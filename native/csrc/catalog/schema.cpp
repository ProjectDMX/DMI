#include "schema.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <thread>

#include "../pack/pack_builder.h"
#include "../common/json.h"

namespace jc = dmi_common;

namespace dmi_catalog {

namespace {

constexpr const char* kFacetDdl[] = {
    "facet_version UInt16 MATERIALIZED 1",
    "element_count UInt64 MATERIALIZED toUInt64(arrayProduct(shape))",
    "tensor_rank UInt8 MATERIALIZED toUInt8(length(shape))",
    "token_span UInt64 MATERIALIZED toUInt64(token_end - token_start)",
    "compression_ratio Float32 MATERIALIZED "
    "toFloat32(if(stored_length = 0, 0, decoded_length / stored_length))",
};

constexpr const char* kCaptureTableOrder =
    "tenant_id, experiment_id, run_id, captured_at_ns, capture_id, "
    "store_id, pack_id";
constexpr const char* kPackTableOrder = "store_id, pack_id";

void split_engine_arguments(const std::string& engine_full,
                            std::string* name_out,
                            std::vector<std::string>* arguments_out) {
  // `engine_full` is the whole engine clause, not just the call: the
  // server renders it as "ReplacingMergeTree(index_version) ORDER BY (...)
  // SETTINGS ...". The arguments end at the parenthesis that CLOSES the
  // engine call — only the first balanced group is read; reading to the
  // last ')' sweeps ORDER BY's own group in and refuses every healthy
  // catalog. A bare call is rendered WITHOUT parentheses (observed on an
  // embedded 26.7), so the name is the leading identifier and arguments
  // are read only when a '(' follows it directly.
  size_t name_end = 0;
  while (name_end < engine_full.size() &&
         ((engine_full[name_end] >= 'a' && engine_full[name_end] <= 'z') ||
          (engine_full[name_end] >= 'A' && engine_full[name_end] <= 'Z') ||
          (engine_full[name_end] >= '0' && engine_full[name_end] <= '9') ||
          engine_full[name_end] == '_')) {
    ++name_end;
  }
  *name_out = engine_full.substr(0, name_end);
  std::string rest = engine_full.substr(name_end);
  size_t at = 0;
  while (at < rest.size() && rest[at] == ' ') ++at;
  if (at >= rest.size() || rest[at] != '(') return;
  rest = rest.substr(at + 1);
  int depth = 1;
  size_t end = rest.size();
  for (size_t i = 0; i < rest.size(); ++i) {
    depth += (rest[i] == '(') - (rest[i] == ')');
    if (depth == 0) {
      end = i;
      break;
    }
  }
  std::string arguments = rest.substr(0, end);
  std::string current;
  for (const char c : arguments) {
    if (c == ',') {
      if (!current.empty()) arguments_out->push_back(current);
      current.clear();
    } else if (c != ' ') {
      current.push_back(c);
    }
  }
  if (!current.empty()) arguments_out->push_back(current);
}

bool keeps_newest_by_index_version(const std::string& engine_full) {
  // The PROPERTY, not the spelling: the Replicated and Shared members of
  // the family carry the version argument LAST, after their replication
  // arguments, and a Replicated-database server converts this build's own
  // DDL into one of them.
  std::string name;
  std::vector<std::string> arguments;
  split_engine_arguments(engine_full, &name, &arguments);
  return (name == "ReplacingMergeTree" ||
          name == "ReplicatedReplacingMergeTree" ||
          name == "SharedReplacingMergeTree") &&
         !arguments.empty() && arguments.back() == "index_version";
}

}  // namespace

CatalogSchema::CatalogSchema(std::shared_ptr<const ClickHouseClient> client,
                             std::string database, std::string prefix)
    : client_(std::move(client)), database_(std::move(database)),
      prefix_(std::move(prefix)) {
  objects_ = {
      {"VIEW", prefix_ + "_capture"},
      {"VIEW", prefix_ + "_pack_inventory"},
      {"TABLE", prefix_ + "_capture_raw"},
      {"TABLE", prefix_ + "_pack_inventory_raw"},
      {"TABLE", prefix_ + "_capture_version_claims"},
      {"TABLE", prefix_ + "_publisher_lease"},
      {"TABLE", prefix_ + "_index_watermark"},
      {"TABLE", prefix_ + "_snapshot_manifest"},
      {"TABLE", prefix_ + "_schema_version"},
  };
  legacy_objects_ = {{"TABLE", prefix_ + "_pack_commit_log"}};
  for (const auto& [kind, name] : objects_) {
    names_[name] = name;
  }
  for (const auto& [kind, name] : legacy_objects_) {
    names_[name] = name;
  }
}

std::string CatalogSchema::qualified(const std::string& table) const {
  return "`" + database_ + "`.`" + table + "`";
}

void CatalogSchema::refuse(const std::string& what) const {
  throw CatalogError(CatalogError::Kind::kSchema, what);
}

std::string CatalogSchema::rebuild_instruction() const {
  std::string objects;
  // `self.objects + self.legacy_objects`, in that order: this build's own
  // first, superseded ones last. Prescribing only objects_ left the
  // legacy object out of the drop list, and an operator who follows the
  // list by name drops what it names, reruns ensure_schema and is refused
  // AGAIN -- by then the list prescribes nine objects that are all
  // already gone, so it names nothing actionable at all. drop() has
  // always looped both, so only the operator working from the message
  // was trapped.
  for (const auto& [kind, name] : objects_) {
    objects += (objects.empty() ? "" : ", ") + qualified(name);
  }
  for (const auto& [kind, name] : legacy_objects_) {
    objects += (objects.empty() ? "" : ", ") + qualified(name);
  }
  return (
      "The catalog is a derived projection over immutable packs, so "
      "rebuilding it loses nothing: stop every indexer, drop ALL of its "
      "objects in this order (views first) -- " + objects + " -- then run "
      "ensure_schema, acquire the publisher lease on the rebuilding writer, "
      "and rebuild over each pack store. Dropping the pack inventory is "
      "mandatory, not optional.");
}

std::vector<SchemaObject> CatalogSchema::catalog_state() const {
  std::string name_list;
  for (const auto& [name, _] : names_) {
    if (!name_list.empty()) name_list += ",";
    name_list += "'" + name + "'";
  }
  const std::vector<Row> rows = client_->execute(
      "SELECT name, engine, sorting_key, engine_full FROM system.tables "
      "WHERE database = '" + database_ + "' AND name IN (" + name_list + ")");
  std::vector<SchemaObject> found;
  for (const Row& row : rows) {
    found.push_back({names_.at(row[0]), row[0], row[1], row[2], row[3]});
  }
  return found;
}

void CatalogSchema::require_catalog_visibility() const {
  // `system.tables` is grant-filtered per role: an object this role holds
  // no privilege on is absent from the state, indistinguishable from one
  // that was dropped. `CHECK GRANT` names an object rather than resolving
  // one, so it needs neither the database nor the objects to exist.
  for (const auto& [kind, name] : objects_) {
    try {
      const std::vector<Row> rows = client_->execute(
          "CHECK GRANT SHOW TABLES ON " + qualified(name));
      if (!rows.empty() && !rows[0].empty() && rows[0][0] != "0") continue;
    } catch (const ClickHouseError&) {
      // Older servers reject CHECK GRANT while parsing; there is no
      // fallback to degrade to, so the requirement is stated.
      refuse("catalog `" + database_ + "`.`" + prefix_ +
             "_*` cannot be checked because this server refused `CHECK "
             "GRANT`, which this build requires. `CHECK GRANT` arrived in "
             "ClickHouse 24.11; upgrade the server to 24.11 or later.");
    }
    refuse("catalog `" + database_ + "`.`" + prefix_ +
           "_*` cannot be checked safely because this role lacks SHOW "
           "TABLES on " + qualified(name) +
           ". Grant catalog visibility and run ensure_schema again.");
  }
}

void CatalogSchema::reject_wrong_kinds(
    const std::vector<SchemaObject>& found) const {
  for (const auto& [kind, name] : objects_) {
    const auto it = std::find_if(found.begin(), found.end(),
                                 [&](const SchemaObject& o) {
                                   return o.name == name;
                                 });
    if (it == found.end()) continue;
    // Views appear in system.tables with engine "View"; everything this
    // build creates as a TABLE must be a non-View, and vice versa.
    const bool is_view = it->engine == "View";
    if ((kind == "VIEW") != is_view) {
      refuse("catalog `" + database_ + "`.`" + prefix_ +
             "_*` holds an object of the wrong kind: `" + name + "` is a " +
             (is_view ? "VIEW" : "TABLE") + " (engine " + it->engine +
             ") where this build creates a " + kind + ".");
    }
  }
}

void CatalogSchema::reject_legacy_objects_beside_this_build(
    const std::vector<SchemaObject>& found) const {
  // Refuse a catalog an EARLIER build has also been writing.
  //
  // A superseded object standing beside this build's own is not a cleanup
  // that was never finished -- the leftovers-only refusal above covers
  // that, where nothing of this build is there. It is two builds sharing
  // one prefix, and the older one is the dangerous half: its ensure_schema
  // is all CREATE ... IF NOT EXISTS, so it no-ops over these tables and
  // recreates its own; its publish writes the pack inventory and an
  // UNCONDITIONAL watermark row carrying no publish identity, and never a
  // manifest row.
  //
  // Nothing else here notices. The stamp reads this version, no table is
  // missing, and the inventory-without-membership check is per-pack, so it
  // reports the older writer's packs only once they are already durable
  // and invisible. Refusing costs a correct deployment nothing: the
  // rebuild instruction lists these objects, so a catalog rebuilt by this
  // build has none of them.
  std::string leftovers;
  for (const auto& [kind, name] : legacy_objects_) {
    if (std::none_of(found.begin(), found.end(), [&](const SchemaObject& o) {
          return o.name == name;
        })) {
      continue;
    }
    if (!leftovers.empty()) leftovers += ", ";
    leftovers += "`" + name + "`";
  }
  if (leftovers.empty()) return;
  refuse("catalog `" + database_ + "`.`" + prefix_ + "_*` is at schema "
         "version " + std::to_string(kSchemaVersion) + " and " + leftovers +
         " stands beside it: an object only an earlier build creates. "
         "Either that build is still writing this prefix -- in which case "
         "its packs are entering the pack inventory with no snapshot "
         "membership, so they are already invisible to every reader and "
         "already skipped by every indexing pass -- or a previous rebuild "
         "left it behind. Stop every writer that is not this build. " +
         rebuild_instruction());
}

void CatalogSchema::reject_wrong_sort_key(
    const std::vector<SchemaObject>& found, const std::string& stamp) const {
  // The layout, not just the inventory: the DDL's own IF NOT EXISTS could
  // no-op against a table another initializer created with a pre-v4 sort
  // key, and the stamp would go on over a layout this build does not
  // describe. Both ReplacingMergeTree tables are checked.
  const std::vector<std::pair<std::string, std::string>> keyed{
      {prefix_ + "_capture_raw", kCaptureTableOrder},
      {prefix_ + "_pack_inventory_raw", kPackTableOrder}};
  for (const auto& [name, order] : keyed) {
    const auto it = std::find_if(
        found.begin(), found.end(),
        [&](const SchemaObject& o) { return o.name == name; });
    if (it == found.end()) continue;
    std::string expected = order;
    std::string actual = it->sorting_key;
    actual.erase(std::remove(actual.begin(), actual.end(), ' '), actual.end());
    expected.erase(std::remove(expected.begin(), expected.end(), ' '),
                   expected.end());
    if (actual != expected) {
      refuse("catalog `" + database_ + "`.`" + prefix_ + "_*` " + stamp +
             " but `" + name + "` sorts by " + it->sorting_key +
             " where this build requires (" + order + "). An ORDER BY "
             "cannot be altered in place. " + rebuild_instruction());
    }
  }
}

void CatalogSchema::reject_wrong_engine(
    const std::vector<SchemaObject>& found, const std::string& stamp) const {
  for (const std::string& name :
       {std::string(prefix_ + "_capture_raw"),
        std::string(prefix_ + "_pack_inventory_raw")}) {
    const auto it = std::find_if(
        found.begin(), found.end(),
        [&](const SchemaObject& o) { return o.name == name; });
    if (it == found.end()) continue;
    if (keeps_newest_by_index_version(it->engine_full)) continue;
    refuse("catalog `" + database_ + "`.`" + prefix_ + "_*` " + stamp +
           " but `" + name + "` is " + it->engine_full +
           " where this build requires ReplacingMergeTree(index_version) "
           "-- or its Replicated/Shared form with index_version as the "
           "version column. Without the version argument a merge that "
           "collapses a duplicate key keeps an arbitrary row rather than "
           "the newest. An engine's arguments cannot be altered in place. " +
           rebuild_instruction());
  }
}

bool CatalogSchema::holds_catalog_data(
    const std::vector<SchemaObject>& found) const {
  // Only the DATA tables are asked: a stamp row, a version claim or a
  // lease row survives an interrupted install by design and hides nothing.
  std::string probes;
  for (const std::string& name :
       {std::string(prefix_ + "_capture_raw"),
        std::string(prefix_ + "_pack_inventory_raw"),
        std::string(prefix_ + "_index_watermark"),
        std::string(prefix_ + "_snapshot_manifest"),
        std::string(prefix_ + "_pack_commit_log")}) {
    if (std::none_of(found.begin(), found.end(), [&](const SchemaObject& o) {
          return o.name == name;
        })) {
      continue;
    }
    if (!probes.empty()) probes += " UNION ALL ";
    probes += "SELECT 1 FROM " + qualified(name) + " LIMIT 1";
  }
  if (probes.empty()) return false;
  const std::vector<Row> rows = client_->execute(
      "SELECT count() FROM (" + probes + ")", {}, deciding_read());
  return !rows.empty() && rows[0][0] != "0";
}

bool CatalogSchema::inventory_without_membership() const {
  const std::vector<Row> rows = client_->execute(
      "SELECT count() FROM (SELECT store_id, pack_id FROM " +
          qualified(prefix_ + "_pack_inventory_raw") + " FINAL "
          "WHERE (store_id, pack_id) NOT IN (SELECT store_id, pack_id FROM " +
          qualified(prefix_ + "_snapshot_manifest") +
          " WHERE (index_version, publish_id) IN (SELECT index_version, "
          "publish_id FROM " +
          qualified(prefix_ + "_index_watermark") + ")) LIMIT 1)",
      {},
      // Deciding: this decides whether ensure_schema REFUSES.
      deciding_read());
  return !rows.empty() && rows[0][0] != "0";
}

std::optional<uint64_t> CatalogSchema::recorded_version() const {
  const std::vector<Row> rows = client_->execute(
      "SELECT version FROM " + qualified(prefix_ + "_schema_version") +
      " ORDER BY version DESC LIMIT 1");
  if (rows.empty()) return std::nullopt;
  return parse_u64_field(rows[0][0], "schema version");
}

CatalogSchema::State CatalogSchema::verify_compatibility_state(
    const std::vector<SchemaObject>& found) const {
  if (found.empty()) return State::kFresh;
  reject_wrong_kinds(found);
  const bool any_object = std::any_of(
      objects_.begin(), objects_.end(), [&](const auto& object) {
        return std::any_of(found.begin(), found.end(), [&](const auto& o) {
          return o.name == object.second;
        });
      });
  if (!any_object) {
    refuse("catalog `" + database_ + "`.`" + prefix_ +
           "_*` holds only leftovers of other builds -- objects this build "
           "does not create beside objects this build requires. " +
           rebuild_instruction());
  }
  const bool stamp_table_present = std::any_of(
      found.begin(), found.end(), [&](const SchemaObject& o) {
        return o.name == prefix_ + "_schema_version";
      });
  if (!stamp_table_present) {
    refuse("catalog `" + database_ + "`.`" + prefix_ +
           "_*` holds catalog tables with no stamp table (an install that "
           "died before the layout, or an older build's): the stamp is the "
           "only record of what the layout is. " + rebuild_instruction());
  }
  const auto recorded = recorded_version();
  if (recorded.has_value() && *recorded != kSchemaVersion) {
    refuse("catalog `" + database_ + "`.`" + prefix_ + "_*` is at schema "
           "version " + std::to_string(*recorded) + " and this build reads "
           "version " + std::to_string(kSchemaVersion) +
           ". A higher version means a newer writer owns this catalog: "
           "upgrade this build rather than writing to it. A lower one is "
           "not upgraded in place. " + rebuild_instruction());
  }
  // What the catalog IS, quoted by every refusal below. A stamp table
  // holding no row is an install of this build that died before stamping,
  // and calling that "stamped" sends the operator looking for a version
  // conflict that is not there.
  const std::string stamp =
      recorded.has_value()
          ? "is stamped schema version " + std::to_string(kSchemaVersion)
          : "holds `" + prefix_ + "_schema_version` with no row in it (an "
            "install of this build that died before stamping)";
  reject_legacy_objects_beside_this_build(found);
  // A superseded object standing beside this build's own is two builds
  // sharing one prefix, and the OLDER one is the dangerous half: its
  // publish writes the pack inventory and an unconditional watermark row
  // carrying no publish identity, and never a manifest row. Nothing else
  // here notices: the stamp reads this version, no table is missing, and
  // the inventory-without-membership check is per-pack.
  for (const auto& [kind, legacy] : legacy_objects_) {
    if (std::none_of(found.begin(), found.end(), [&](const SchemaObject& o) {
          return o.name == legacy;
        })) {
      continue;
    }
    refuse("catalog `" + database_ + "`.`" + prefix_ + "_*` " + stamp +
           " and `" + legacy +
           "` stands beside it: an object only an earlier build creates. "
           "Either that build is still writing this prefix -- in which "
           "case its packs are entering the pack inventory with no "
           "snapshot membership, so they are already invisible to every "
           "reader and already skipped by every indexing pass -- or a "
           "previous rebuild left it behind. Stop every writer that is "
           "not this build. " + rebuild_instruction());
  }
  std::vector<std::string> missing;
  for (const auto& [kind, name] : objects_) {
    if (kind != "TABLE") continue;
    if (std::none_of(found.begin(), found.end(), [&](const SchemaObject& o) {
          return o.name == name;
        })) {
      missing.push_back(name);
    }
  }
  if (!missing.empty() && !holds_catalog_data(found)) {
    // Nothing survived that recreating them empty could hide, so this is
    // an unfinished install and the DDL completes it — the reason the
    // stamp table is created FIRST.
    return State::kIncomplete;
  }
  if (!missing.empty()) {
    std::string list;
    for (const std::string& name : missing) {
      list += (list.empty() ? "" : ", ") + qualified(name);
    }
    refuse("catalog `" + database_ + "`.`" + prefix_ + "_*` " + stamp +
           " but is missing " + list +
           ". Recreating those empty beside tables that kept their rows is "
           "not a repair: surviving inventory would skip their packs. " +
           rebuild_instruction());
  }
  reject_wrong_sort_key(found, stamp);
  reject_wrong_engine(found, stamp);
  if (inventory_without_membership()) {
    refuse("catalog `" + database_ + "`.`" + prefix_ +
           "_*` lists one or more packs in the inventory without membership "
           "admitted by a publish. Those packs are already marked indexed "
           "but belong to no snapshot, so an indexing pass would skip them "
           "and report success while their captures remain invisible. " +
           rebuild_instruction());
  }
  return State::kComplete;
}

std::string CatalogSchema::verify_compatibility() const {
  const std::vector<SchemaObject> found = catalog_state();
  switch (verify_compatibility_state(found)) {
    case State::kFresh: return "fresh";
    case State::kIncomplete: return "incomplete";
    case State::kComplete: return "complete";
  }
  return "unknown";
}

void CatalogSchema::create_stamp_and_lease_tables() const {
  client_->execute(
      "CREATE TABLE IF NOT EXISTS " + qualified(prefix_ + "_schema_version") +
      " (\nversion UInt32, applied_at_ns UInt64\n) ENGINE = MergeTree ORDER "
      "BY version");
  client_->execute(
      "CREATE TABLE IF NOT EXISTS " + qualified(prefix_ + "_publisher_lease") +
      " (\nterm UInt64, lease_id UUID, holder String, acquired_at_ns UInt64,\n"
      "expires_at_ns UInt64\n) ENGINE = MergeTree ORDER BY (term, lease_id)");
}

void CatalogSchema::lay_out() const {
  const std::string capture_raw = qualified(prefix_ + "_capture_raw");
  std::string facets;
  for (const char* facet : kFacetDdl) {
    facets += std::string(",\n") + facet;
  }
  client_->execute(
      "CREATE TABLE IF NOT EXISTS " + capture_raw +
      " (\ncapture_id String, tenant_id String, experiment_id String, "
      "run_id String,\nsession_id String, request_id String, sequence_id "
      "String, model_id String,\nmodel_revision String, adapter_revision "
      "Nullable(String),\ncapture_policy_version String, hook_name "
      "LowCardinality(String), layer_number Int32,\nproducer_rank UInt32, "
      "step_number UInt64, token_start UInt64, token_end UInt64,\n"
      "batch_position UInt32, dtype LowCardinality(String), shape "
      "Array(UInt32),\ncaptured_at_ns UInt64, pack_id UUID, store_id "
      "LowCardinality(String), object_key String,\nobject_bytes UInt64, "
      "pack_checksum FixedString(64), pack_record_count UInt32,\n"
      "payload_offset UInt64, stored_length UInt64, decoded_length UInt64,\n"
      "codec LowCardinality(String), payload_checksum FixedString(8), "
      "index_version UInt64" +
      facets + "\n) ENGINE = ReplacingMergeTree(index_version)\nORDER BY (" +
      kCaptureTableOrder + ")");
  client_->execute(
      "CREATE TABLE IF NOT EXISTS " + qualified(prefix_ + "_pack_inventory_raw") +
      " (\npack_id UUID, store_id LowCardinality(String), object_key String,\n"
      "object_bytes UInt64, pack_checksum FixedString(64), record_count UInt32,\n"
      "index_version UInt64\n) ENGINE = ReplacingMergeTree(index_version)\n"
      "ORDER BY (" + kPackTableOrder + ")");
  for (const char* facet : kFacetDdl) {
    client_->execute("ALTER TABLE " + capture_raw +
                     " ADD COLUMN IF NOT EXISTS " + std::string(facet));
  }
  client_->execute("ALTER TABLE " + capture_raw +
                   " ADD INDEX IF NOT EXISTS capture_id_bloom capture_id "
                   "TYPE bloom_filter(0.01) GRANULARITY 4");
  client_->execute("ALTER TABLE " + capture_raw +
                   " MATERIALIZE INDEX capture_id_bloom");
  client_->execute(
      "CREATE TABLE IF NOT EXISTS " + qualified(prefix_ + "_index_watermark") +
      " (\nindex_version UInt64, publish_id UUID, published_at_ns UInt64,\n"
      "indexed_rows UInt64, indexed_packs UInt32\n) ENGINE = MergeTree "
      "ORDER BY (index_version, publish_id)");
  client_->execute(
      "CREATE TABLE IF NOT EXISTS " +
      qualified(prefix_ + "_capture_version_claims") +
      " (\nversion UInt64, claim_id UUID, claimed_at_ns UInt64\n) ENGINE = "
      "MergeTree ORDER BY (version, claim_id)");
  client_->execute(
      "CREATE TABLE IF NOT EXISTS " + qualified(prefix_ + "_snapshot_manifest") +
      " (\nindex_version UInt64, publish_id UUID, store_id "
      "LowCardinality(String),\npack_id UUID\n) ENGINE = MergeTree ORDER BY "
      "(index_version, publish_id, store_id, pack_id)");
  // The views, definitions shared with the quorum verifier.
  client_->execute(
      "CREATE OR REPLACE VIEW " + qualified(prefix_ + "_capture") + " AS "
      "SELECT capture_id, tenant_id, experiment_id, run_id, session_id, "
      "request_id, sequence_id, model_id, model_revision, "
      "adapter_revision, capture_policy_version, hook_name, layer_number, "
      "producer_rank, step_number, token_start, token_end, batch_position, "
      "dtype, shape, captured_at_ns, pack_id, store_id, object_key, "
      "object_bytes, pack_checksum, pack_record_count, payload_offset, "
      "stored_length, decoded_length, codec, payload_checksum "
      "FROM " + qualified(prefix_ + "_capture_raw") + " FINAL "
      "WHERE (store_id, pack_id) IN ("
      "SELECT store_id, pack_id FROM `" + database_ + "`.`" + prefix_ +
      "_snapshot_manifest` WHERE (index_version, publish_id) IN "
      "(SELECT index_version, publish_id FROM `" + database_ + "`.`" +
      prefix_ + "_index_watermark`))");
  client_->execute(
      "CREATE VIEW IF NOT EXISTS " + qualified(prefix_ + "_pack_inventory") +
      " AS SELECT pack_id, store_id, object_key, object_bytes, "
      "pack_checksum, record_count FROM " +
      qualified(prefix_ + "_pack_inventory_raw") + " FINAL");
}

void CatalogSchema::stamp() const {
  const auto now = std::chrono::system_clock::now();
  const uint64_t applied_at_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch())
          .count());
  client_->execute(
      "INSERT INTO " + qualified(prefix_ + "_schema_version") +
      " (version, applied_at_ns) "
      "SELECT toUInt32(%(version)s), toUInt64(%(applied_at_ns)s) "
      "FROM system.one WHERE (SELECT count() FROM " +
      qualified(prefix_ + "_schema_version") + ") = 0",
      {{"version", kSchemaVersion}, {"applied_at_ns", applied_at_ns}});
}

void CatalogSchema::confirm_catalog_is_complete() const {
  const std::vector<SchemaObject> found = catalog_state();
  for (const auto& [kind, name] : objects_) {
    if (std::none_of(found.begin(), found.end(), [&](const SchemaObject& o) {
          return o.name == name;
        })) {
      refuse("catalog `" + database_ + "`.`" + prefix_ +
             "_*` is incomplete after schema creation (not visible: `" +
             name + "`). It was not stamped.");
    }
  }
  reject_wrong_sort_key(found, "was just created by this build");
  reject_wrong_engine(found, "was just created by this build");
}

bool CatalogSchema::take_the_install_lease(LeaseCoordinator* leases,
                                           uint64_t retry_sleep_ns) const {
  // Wait for the install lease; false means somebody else finished it.
  // A cold fleet start has every process installing at once, and refusing
  // outright made all but one of them crash out of a call every caller
  // treats as infallible setup.
  const double budget_s =
      static_cast<double>(leases->ttl_ns()) / 1e9 + kInstallLeaseMarginS;
  const int attempts =
      std::max(1, static_cast<int>(std::ceil(budget_s / kInstallLeaseRetryS)));
  for (int attempt = 0; attempt < attempts; ++attempt) {
    try {
      leases->acquire("ensure_schema");
      return true;
    } catch (const CatalogError& e) {
      if (e.kind() != CatalogError::Kind::kHeld) throw;
      if (verify_compatibility() == "complete") {
        create_stamp_and_lease_tables();
        lay_out();
        confirm_catalog_is_complete();
        stamp();
        return false;
      }
      if (attempt + 1 == attempts) throw;
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(retry_sleep_ns));
    }
  }
  throw CatalogError(CatalogError::Kind::kSchema,
                     "unreachable: the last attempt re-raises");
}

void CatalogSchema::ensure(LeaseCoordinator* leases, uint64_t retry_sleep_ns) {
  const std::vector<SchemaObject> found = catalog_state();
  // Visibility before any VERDICT: every refusal below is read off a
  // grant-filtered system.tables, so an ungranted object would otherwise
  // be diagnosed as missing and prescribe the teardown of a healthy
  // catalog.
  require_catalog_visibility();
  const State state = verify_compatibility_state(found);
  client_->execute("CREATE DATABASE IF NOT EXISTS `" + database_ + "`");
  if (state == State::kComplete) {
    // Nothing to serialise: the DDL is idempotent over it and the stamp
    // is a server-side no-op. Taking the lease here would refuse every
    // second process for as long as an indexer is publishing.
    create_stamp_and_lease_tables();
    lay_out();
    confirm_catalog_is_complete();
    stamp();
    return;
  }
  // The two tables an installer needs to take the lease are created
  // first, both idempotent; everything after them is under the lease.
  create_stamp_and_lease_tables();
  const bool borrowed = leases->lease() != nullptr;
  if (borrowed) {
    leases->renew();
  } else if (!take_the_install_lease(leases, retry_sleep_ns)) {
    // Another initialiser finished while this one waited.
    return;
  }
  try {
    // Re-read under the lease: what looked like nothing may now be
    // another initialiser's finished or half-finished work.
    verify_compatibility();
    lay_out();
    confirm_catalog_is_complete();
    stamp();
  } catch (...) {
    if (!borrowed) {
      try {
        leases->release();
      } catch (...) {
      }
    }
    throw;
  }
  if (!borrowed) leases->release();
}

void CatalogSchema::drop() const {
  // DROP TABLE for all of them, because the catalog that most needs
  // tearing down is the one whose kinds are WRONG. Views still go first.
  for (const auto& [kind, name] : objects_) {
    client_->execute("DROP TABLE IF EXISTS " + qualified(name));
  }
  for (const auto& [kind, name] : legacy_objects_) {
    client_->execute("DROP TABLE IF EXISTS " + qualified(name));
  }
}

}  // namespace dmi_catalog

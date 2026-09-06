// B4: the schema port — ensure/drop from clickhouse_schema.py. The DDL is
// the Python module's, textually. The install is serialised on the
// publisher lease (the catalog's one mutual-exclusion primitive), the
// stamp is a server-side conditional INSERT, and compatibility is refused
// rather than repaired: kinds, sort keys, the ReplacingMergeTree version
// argument (a property of engine_full, not a spelling), version stamp,
// and inventory-without-membership. The derivations live in the Python
// module; this port keeps the statements, the order, and the refusals.

#ifndef DMI_CATALOG_SCHEMA_H
#define DMI_CATALOG_SCHEMA_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "clickhouse_client.h"
#include "lease_coordinator.h"

namespace dmi_catalog {

constexpr int kSchemaVersion = 4;

// The install-lease wait: one TTL plus slack, counted in attempts so a
// test can drive the budget with a sleep that does not sleep.
constexpr double kInstallLeaseRetryS = 0.5;
constexpr double kInstallLeaseMarginS = 5.0;

struct SchemaObject {
  std::string kind;  // "TABLE" or "VIEW"
  std::string name;
  std::string engine;
  std::string sorting_key;
  std::string engine_full;
};

class CatalogSchema {
 public:
  CatalogSchema(std::shared_ptr<const ClickHouseClient> client,
                std::string database, std::string prefix);

  // Create or verify the catalog; installs are serialised per prefix.
  // `leases` is the calling writer's coordinator: a writer that already
  // holds the lease keeps it and renews it around the install; otherwise
  // the install takes the lease itself and gives it back at the stamp.
  // retry_sleep_ns drives the wait between install-lease attempts.
  void ensure(LeaseCoordinator* leases, uint64_t retry_sleep_ns);
  void drop() const;

  // FRESH / INCOMPLETE / COMPLETE as "fresh" / "incomplete" / "complete";
  // anything incompatible throws CatalogError (kSchema).
  std::string verify_compatibility() const;

 private:
  enum class State { kFresh, kIncomplete, kComplete };

  std::string qualified(const std::string& table) const;
  std::vector<SchemaObject> catalog_state() const;
  void require_catalog_visibility() const;
  State verify_compatibility_state(
      const std::vector<SchemaObject>& found) const;
  void reject_wrong_kinds(const std::vector<SchemaObject>& found) const;
  void reject_wrong_sort_key(const std::vector<SchemaObject>& found,
                             const std::string& stamp) const;
  void reject_wrong_engine(const std::vector<SchemaObject>& found,
                           const std::string& stamp) const;
  void confirm_catalog_is_complete() const;
  void create_stamp_and_lease_tables() const;
  void lay_out() const;
  void stamp() const;
  bool holds_catalog_data(const std::vector<SchemaObject>& found) const;
  bool inventory_without_membership() const;
  std::optional<uint64_t> recorded_version() const;
  bool take_the_install_lease(LeaseCoordinator* leases,
                              uint64_t retry_sleep_ns) const;
  [[noreturn]] void refuse(const std::string& what) const;
  std::string rebuild_instruction() const;

  std::shared_ptr<const ClickHouseClient> client_;
  std::string database_;
  std::string prefix_;
  std::vector<std::pair<std::string, std::string>> objects_;
  std::vector<std::pair<std::string, std::string>> legacy_objects_;
  std::map<std::string, std::string> names_;  // short name -> member name
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_SCHEMA_H

/*
 * Architecture renderer.
 *
 * Draws the diagram from the `architecture_layout` payload the server sends --
 * no hardcoded model structure lives here. Every node becomes a real <g>
 * element carrying `data-node-id`, so selection, hover, and state styling are
 * ordinary DOM concerns.
 */
(function (global) {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  var W = 420;
  var PAD = 14;
  var GLOBAL_H = 48;
  var CHILD_H = 46;
  var CHILD_GAP = 8;
  var STACK_HEADER = 32;
  var ARROW = 26;
  var GHOSTS = 2;
  var GHOST_STEP = 5;

  function el(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    for (var key in attrs || {}) {
      if (Object.prototype.hasOwnProperty.call(attrs, key)) {
        node.setAttribute(key, attrs[key]);
      }
    }
    return node;
  }

  function text(content, attrs) {
    var node = el("text", attrs);
    node.textContent = content;
    return node;
  }

  /* normal | selected | partial | unavailable, from how many hooks are on. */
  function nodeState(node, selected) {
    if (!node.available) return "unavailable";
    var usable = node.hooks.filter(function (hook) { return hook.available; });
    if (!usable.length) return "unavailable";
    var on = usable.filter(function (hook) { return selected.has(hook.id); }).length;
    if (on === 0) return "normal";
    return on === usable.length ? "selected" : "partial";
  }

  function countLabel(node, selected) {
    var usable = node.hooks.filter(function (hook) { return hook.available; });
    var on = usable.filter(function (hook) { return selected.has(hook.id); }).length;
    return on ? on + " / " + usable.length : "";
  }

  function drawArrow(parent, x, y, height) {
    parent.appendChild(el("path", {
      class: "stack-edge",
      d: "M " + x + " " + y + " L " + x + " " + (y + height - 7)
    }));
    parent.appendChild(el("path", {
      class: "stack-arrow",
      d: "M " + (x - 4) + " " + (y + height - 8) +
         " L " + (x + 4) + " " + (y + height - 8) +
         " L " + x + " " + (y + height) + " Z"
    }));
  }

  function drawNode(parent, node, box, state, count, handlers) {
    var group = el("g", {
      class: "node",
      "data-node-id": node.id,
      "data-state": state,
      "data-selectable": state === "unavailable" ? "false" : "true",
      "data-focused": handlers.focused === node.id ? "true" : "false",
      tabindex: state === "unavailable" ? "-1" : "0",
      role: "button",
      "aria-label": node.label + " observations"
    });

    group.appendChild(el("rect", {
      class: "node-box",
      x: box.x, y: box.y, width: box.w, height: box.h, rx: 9
    }));

    var hasSub = !!box.sub;
    group.appendChild(text(node.label, {
      class: "node-label",
      x: box.x + 14,
      y: box.y + (hasSub ? box.h / 2 - 2 : box.h / 2 + 5)
    }));

    if (hasSub) {
      group.appendChild(text(box.sub, {
        class: "node-sub",
        x: box.x + 14,
        y: box.y + box.h / 2 + 14
      }));
    }

    if (count) {
      group.appendChild(text(count, {
        class: "node-count",
        x: box.x + box.w - 14,
        y: box.y + box.h / 2 + 4,
        "text-anchor": "end"
      }));
    }

    if (state !== "unavailable") {
      group.addEventListener("click", function () { handlers.onSelect(node.id); });
      group.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handlers.onSelect(node.id);
        }
      });
    }

    parent.appendChild(group);
  }

  function hookSummary(node) {
    return node.hooks.map(function (hook) { return hook.id; }).join(" · ");
  }

  /**
   * Render into `container`.
   *
   * @param {Element} container
   * @param {Object}  layout     architecture_layout payload from GET /api/model
   * @param {Object}  options    {selected: Set, focused: string|null, onSelect: fn}
   */
  function render(container, layout, options) {
    var selected = options.selected || new Set();
    var handlers = {
      focused: options.focused || null,
      onSelect: options.onSelect || function () {}
    };

    var globals = layout.nodes.filter(function (n) { return n.scope === "global"; });
    var perLayer = layout.nodes.filter(function (n) { return n.scope !== "global"; });

    // Convention: the first global node is the input side, the rest tail the
    // stack. Anything else would need the payload to say so explicitly.
    var head = globals.slice(0, 1);
    var tail = globals.slice(1);

    var stackH = STACK_HEADER + perLayer.length * CHILD_H +
                 Math.max(0, perLayer.length - 1) * CHILD_GAP + 14;
    var totalH = PAD +
                 head.length * (GLOBAL_H + ARROW) +
                 (GHOSTS * GHOST_STEP) + stackH + ARROW +
                 tail.length * GLOBAL_H +
                 Math.max(0, tail.length - 1) * ARROW + PAD;

    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + totalH,
      xmlns: SVG_NS,
      role: "group",
      "aria-label": "Model architecture"
    });

    var midX = W / 2;
    var y = PAD;

    head.forEach(function (node) {
      drawNode(svg, node, {
        x: 60, y: y, w: W - 120, h: GLOBAL_H, sub: hookSummary(node)
      }, nodeState(node, selected), countLabel(node, selected), handlers);
      y += GLOBAL_H;
      drawArrow(svg, midX, y, ARROW);
      y += ARROW;
    });

    // Ghost cards behind the layer box carry the "x N" idea visually.
    for (var g = GHOSTS; g >= 1; g -= 1) {
      svg.appendChild(el("rect", {
        class: "node-box",
        x: 26 + g * GHOST_STEP,
        y: y + (GHOSTS - g) * GHOST_STEP,
        width: W - 52,
        height: stackH,
        rx: 11,
        opacity: 0.4 / g
      }));
    }
    y += GHOSTS * GHOST_STEP;

    svg.appendChild(el("rect", {
      class: "node-box",
      x: 26, y: y, width: W - 52, height: stackH, rx: 11
    }));
    svg.appendChild(text("Transformer layer", {
      class: "node-label", x: 42, y: y + 21
    }));
    svg.appendChild(text("x " + layout.num_layers, {
      class: "stack-note", x: W - 42, y: y + 21, "text-anchor": "end"
    }));

    var childY = y + STACK_HEADER;
    perLayer.forEach(function (node) {
      drawNode(svg, node, {
        x: 42, y: childY, w: W - 84, h: CHILD_H, sub: hookSummary(node)
      }, nodeState(node, selected), countLabel(node, selected), handlers);
      childY += CHILD_H + CHILD_GAP;
    });

    y += stackH;
    drawArrow(svg, midX, y, ARROW);
    y += ARROW;

    tail.forEach(function (node, index) {
      if (index > 0) {
        drawArrow(svg, midX, y, ARROW);
        y += ARROW;
      }
      drawNode(svg, node, {
        x: 60, y: y, w: W - 120, h: GLOBAL_H, sub: hookSummary(node)
      }, nodeState(node, selected), countLabel(node, selected), handlers);
      y += GLOBAL_H;
    });

    container.replaceChildren(svg);
  }

  global.DMIArchitecture = { render: render, nodeState: nodeState };
})(window);

// Museum of AI Art — front-end.
// Reads exhibit.json (a static, build-time generated index) and renders the
// gallery wall. No frameworks, no build step.

(function () {
  "use strict";

  const featuredEl = document.getElementById("featured");
  const gridEl = document.getElementById("grid");
  const countEl = document.getElementById("collection-count");

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === "class") node.className = attrs[k];
        else if (k === "html") node.innerHTML = attrs[k];
        else if (k.startsWith("on") && typeof attrs[k] === "function") {
          node.addEventListener(k.slice(2), attrs[k]);
        } else {
          node.setAttribute(k, attrs[k]);
        }
      }
    }
    if (children) {
      for (const c of children) {
        if (c == null) continue;
        node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      }
    }
    return node;
  }

  function escapeHTML(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function romanize(year) {
    if (year == null) return "";
    year = parseInt(year, 10);
    if (!year || year < 1) return "";
    const map = [
      ["M", 1000], ["CM", 900], ["D", 500], ["CD", 400],
      ["C", 100], ["XC", 90], ["L", 50], ["XL", 40],
      ["X", 10], ["IX", 9], ["V", 5], ["IV", 4], ["I", 1],
    ];
    let n = year, out = "";
    for (const [r, v] of map) {
      while (n >= v) { out += r; n -= v; }
    }
    return out;
  }

  function loadIndex() {
    return fetch("exhibit.json", { cache: "no-cache" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
  }

  function renderFeatured(p) {
    const year = p.year;
    const rom = romanize(year);
    const dateStr = p.date;
    const event = p.event || {};
    const sourceLink = event.wikipedia_url
      ? '<a href="' + escapeHTML(event.wikipedia_url) + '" target="_blank" rel="noopener">Wikipedia</a>'
      : "<em>source unrecorded</em>";

    featuredEl.innerHTML = "";
    featuredEl.appendChild(
      el("div", { class: "frame" }, [
        el("img", { src: p.thumbnail, alt: p.title, loading: "eager" }),
      ])
    );
    featuredEl.appendChild(
      el("div", { class: "card-detail" }, [
        el("p", { class: "card-date" }, [dateStr]),
        el("h3", null, [p.title]),
        rom ? el("p", { class: "year" }, ["Anno " + rom]) : null,
        p.medium ? el("p", { class: "medium" }, [p.medium]) : null,
        el("p", { class: "statement" }, [p.artist_statement || p.excerpt || ""]),
        el("div", { class: "source", html: "Responding to: " + escapeHTML(event.text || "") + " (" + sourceLink + ")" }),
      ])
    );
  }

  function renderGrid(items) {
    gridEl.innerHTML = "";
    items.forEach(function (p) {
      const card = el("article", { class: "card" }, [
        el("a", { href: p.url }, [
          el("div", { class: "frame" }, [
            el("img", { src: p.thumbnail, alt: p.title, loading: "lazy" }),
          ]),
          el("div", { class: "meta" }, [
            el("p", { class: "card-date" }, [p.date]),
            el("h3", { class: "card-title" }, [p.title]),
            p.year != null ? el("p", { class: "card-year" }, [String(p.year)]) : null,
            el("p", { class: "card-excerpt" }, [p.excerpt || ""]),
          ]),
        ]),
      ]);
      gridEl.appendChild(card);
    });
  }

  // Featured view needs full details; grid only needs excerpt.
  // To keep exhibit.json small, we re-fetch the per-day meta.json for the
  // most recent painting. Older entries are summarised in the index.
  function loadFeaturedFull(date) {
    return fetch("exhibits/" + date + "/meta.json", { cache: "no-cache" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  loadIndex().then(function (items) {
    if (!items || items.length === 0) {
      featuredEl.innerHTML = '<p class="loading">The collection is being prepared. Check back tomorrow.</p>';
      return;
    }
    countEl.textContent = items.length + (items.length === 1 ? " painting" : " paintings");
    const newest = items[0];
    loadFeaturedFull(newest.date).then(function (full) {
      renderFeatured(Object.assign({}, newest, full || {}));
    });
    renderGrid(items);
  }).catch(function (err) {
    featuredEl.innerHTML = '<p class="loading">The curator could not be reached. (' + escapeHTML(err.message) + ')</p>';
  });
})();

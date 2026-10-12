// Copies each footnote into a <span class="sidenote"> right after its reference,
// so wide screens can show it in the margin. The footnotes section at the
// bottom stays in place for narrow screens; CSS decides which one is visible.
export default function rehypeSidenotes() {
  return (tree) => {
    const notes = new Map();
    collectNotes(tree, notes);
    if (notes.size) insertSidenotes(tree, notes, new Set());
  };
}

// A note can't float out of a table cell, so notes referenced inside these
// go just before the element instead, level with its top edge.
const unfloatable = new Set(['table']);

const classes = (node) => [].concat(node.properties?.className ?? []);

function collectNotes(node, notes) {
  if (node.type === 'element' && node.tagName === 'section' && classes(node).includes('footnotes')) {
    walk(node, (el) => {
      if (el.tagName === 'li' && el.properties?.id) notes.set(el.properties.id, el);
    });
    return;
  }
  node.children?.forEach((child) => collectNotes(child, notes));
}

function insertSidenotes(node, notes, seen) {
  if (!node.children) return;
  if (node.type === 'element' && node.tagName === 'section' && classes(node).includes('footnotes')) return;

  for (let i = 0; i < node.children.length; i++) {
    const child = node.children[i];
    if (child.type !== 'element') continue;

    if (unfloatable.has(child.tagName)) {
      const before = [];
      walk(child, (el) => {
        const ref = el.tagName === 'sup' && findRef(el);
        const made = ref && makeSidenote(ref, notes, seen);
        if (made) before.push(made);
      });
      node.children.splice(i, 0, ...before);
      i += before.length;
      continue;
    }

    const ref = child.tagName === 'sup' && findRef(child);
    if (ref) {
      const made = makeSidenote(ref, notes, seen);
      if (made) node.children.splice(++i, 0, made);
    } else {
      insertSidenotes(child, notes, seen);
    }
  }
}

// Only the first reference to a footnote gets a sidenote.
function makeSidenote(ref, notes, seen) {
  const id = decodeURIComponent(String(ref.properties.href).slice(1));
  const note = notes.get(id);
  if (!note || seen.has(id)) return null;
  seen.add(id);
  return sidenote(textOf(ref), note);
}

function findRef(sup) {
  return sup.children.find(
    (c) => c.type === 'element' && c.tagName === 'a' && String(c.properties?.href ?? '').startsWith('#')
  );
}

// A footnote's paragraphs become block-level spans so they can live inside a <p>.
function sidenote(label, note) {
  const paragraphs = note.children
    .filter((c) => c.type === 'element' && c.tagName === 'p')
    .map((p, i) => ({
      type: 'element',
      tagName: 'span',
      properties: { className: ['sidenote-p'] },
      children: [
        ...(i === 0 ? [span('sidenote-number', label)] : []),
        ...structuredClone(p.children).filter((c) => !isBackref(c)),
      ],
    }));

  return {
    type: 'element',
    tagName: 'span',
    properties: { className: ['sidenote'], role: 'note' },
    children: paragraphs,
  };
}

function isBackref(node) {
  return node.type === 'element' && node.tagName === 'a' && 'dataFootnoteBackref' in (node.properties ?? {});
}

function span(className, text) {
  return { type: 'element', tagName: 'span', properties: { className: [className] }, children: [{ type: 'text', value: text }] };
}

function textOf(node) {
  if (node.type === 'text') return node.value;
  return (node.children ?? []).map(textOf).join('');
}

function walk(node, fn) {
  if (node.type === 'element') fn(node);
  node.children?.forEach((child) => walk(child, fn));
}

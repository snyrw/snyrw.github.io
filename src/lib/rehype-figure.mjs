// Turns a paragraph containing only an image into <figure>.
// If the image has a title — ![alt](src "caption") — it becomes the caption.
export default function rehypeFigure() {
  return (tree) => visit(tree);
}

function visit(node) {
  if (!node.children) return;
  node.children = node.children.map((child) => {
    if (child.type === 'element' && child.tagName === 'p') {
      const content = child.children.filter(
        (c) => !(c.type === 'text' && c.value.trim() === '')
      );
      if (content.length === 1 && content[0].tagName === 'img') {
        const img = content[0];
        const caption = img.properties.title;
        delete img.properties.title;
        const children = [img];
        if (caption) {
          children.push({
            type: 'element',
            tagName: 'figcaption',
            properties: {},
            children: [{ type: 'text', value: caption }],
          });
        }
        return { type: 'element', tagName: 'figure', properties: {}, children };
      }
    }
    visit(child);
    return child;
  });
}

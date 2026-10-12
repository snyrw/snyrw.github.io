# newsite

```
npm install
npm run dev        # http://localhost:4321
```

## Writing

- **Post:** make a folder in `src/content/writing/` with an `index.md` (or `.mdx`) and any images beside it.
  ```
  ---
  title: Post title
  date: 2026-10-01
  description: Optional, used for the RSS feed and link previews.
  draft: true   # optional; drafts show in dev but aren't published
  ---
  ```
  An image on its own line becomes a figure: `![alt text](./image.png "Optional caption")`.

  Footnotes (`text[^1]` … `[^1]: the note`) show in the right margin on wide screens and at the
  bottom of the post otherwise. Keep them to text, links and emphasis; lists or code inside a
  footnote only appear in the bottom version.
- **Note:** add a Markdown file to `src/content/notes/` with a `date` and optional `title`.
- **Name, blurb, links:** `src/site.ts` and `src/pages/index.astro`.

## Layout

```
src/
  content/          posts and notes (Markdown)
  content.config.ts frontmatter fields for each
  pages/            one file per route: home, writing, notes, rss, 404
  layouts/          the page shell (header, footer, theme toggle)
  components/       the post list
  lib/              helpers, plus the Markdown plugins for figures and margin notes
  styles/           the one stylesheet
public/             copied as-is (favicon)
```

If you change a plugin in `src/lib/`, stop the dev server, delete `.astro/`, and start it again;
the running server keeps the old version and cached renders.

## Deploying

Pushing to `main` builds and deploys to GitHub Pages. One-time setup: in the repo's
Settings → Pages, set **Source** to **GitHub Actions**.

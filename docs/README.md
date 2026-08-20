# nobg docs

The documentation site for [nobg](https://github.com/feyninc/nobg), built with
[Fumadocs](https://fumadocs.dev) on Next.js.

## Develop

```bash
npm install
npm run dev
```

Then open http://localhost:3000.

| Script                 | What it does                            |
| ---------------------- | --------------------------------------- |
| `npm run dev`          | Dev server with hot reload              |
| `npm run build`        | Production build                        |
| `npm run start`        | Serve the production build              |
| `npm run types:check`  | `next typegen` + `tsc --noEmit`         |

## Writing content

Pages are MDX files under `content/docs/`. The URL follows the file path, so
`content/docs/guides/onnx.mdx` becomes `/docs/guides/onnx`.

```mdx
---
title: Page title
description: One line, used for search results and OG images.
icon: Package
---
```

`icon` is any [lucide](https://lucide.dev/icons) export name, resolved by `lucideIconsPlugin` in
`lib/source.ts`.

Navigation order comes from the `meta.json` beside the pages — a page not listed there still renders,
it just falls to the end of the sidebar:

```json
{
  "title": "Guides",
  "pages": ["background-removal", "text-prompts", "---Section---", "onnx"]
}
```

`Card`, `Cards`, `Callout` and the code-block components are available in every page without importing.
`Tabs`, `Tab`, `Steps`, `Step`, `TypeTable`, `Accordion` and `Accordions` are registered in
`components/mdx.tsx` — add anything else there rather than importing it per page.

## Layout

| Path                      | What it holds                                                    |
| ------------------------- | ---------------------------------------------------------------- |
| `content/docs/`           | Every documentation page, as MDX                                 |
| `lib/shared.ts`           | Site name, description, repo coordinates, external links         |
| `lib/layout.shared.tsx`   | Nav title, logo and top-level nav links                          |
| `lib/source.ts`           | Content source adapter and page-URL helpers                      |
| `app/(home)/page.tsx`     | Landing page                                                     |
| `app/docs/`               | Docs layout and the catch-all page route                         |
| `app/api/search/route.ts` | Search endpoint (Orama, built at request time from the content)   |
| `app/og/`                 | Generated Open Graph images                                      |

`lib/shared.ts` is the one file to edit when the repo moves or a link changes: `gitConfig.contentDir`
is what makes each page's "Edit on GitHub" link resolve, given the Next app is nested in `docs/`.

## Deploying to Vercel

Import the `nobg` repo and set **Root Directory** to `docs`. Everything else is detected. Pushing to
`main` deploys; every other branch gets a preview.

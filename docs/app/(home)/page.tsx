import Link from 'next/link';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';
import { links } from '@/lib/shared';

const quickStart = `from nobg import AutoModel

model = AutoModel.from_pretrained("feyninc/FeyNobg")
model.process("input.jpg").save("output.png")`;

const features = [
  {
    title: 'One call, no boilerplate',
    body: 'process() loads, preprocesses, runs under no_grad, post-processes and composites. Paths, URLs, base64, numpy arrays and PIL images all work as input.',
  },
  {
    title: 'Two models, one contract',
    body: 'FeyNobg (BiRefNet) for hair-level edges, MultiMatte (SAM3) for text-promptable cutouts. Both return (B, 1, H, W) matte logits, so they are interchangeable.',
  },
  {
    title: 'Hub-native',
    body: 'Built on PyTorchModelHubMixin: from_pretrained, push_to_hub and generated model cards, plus an AutoModel / AutoProcessor pair that dispatch on repo tags.',
  },
  {
    title: 'Trainable',
    body: 'The processor emits labels, the model returns loss, and the criterion is a swappable function attribute — so the HuggingFace Trainer works out of the box.',
  },
  {
    title: 'Halo-free compositing',
    body: 'refine_foreground solves for unmixed foreground colors in pure torch, on whatever device the tensors already live on.',
  },
  {
    title: 'ONNX export',
    body: 'onnx_save_pretrained / onnx_from_pretrained mirror the Hub methods, and the returned graph keeps the same process / predict API.',
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      <section className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-4 py-16 md:py-24">
        <div className="flex flex-col items-start gap-6">
          <span className="rounded-full border px-3 py-1 text-xs text-fd-muted-foreground">
            Apache-2.0 · Python ≥ 3.10 · torch ≥ 2.0
          </span>
          <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
            Background removal that stops at the hair.
          </h1>
          <p className="max-w-2xl text-fd-muted-foreground md:text-lg">
            <span className="font-medium text-fd-foreground">nobg</span> is an open-source library
            for background removal and image matting. Matting models, image processors, losses and
            metrics behind one small API — with the HuggingFace Hub on both ends.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/docs"
              className="rounded-lg bg-fd-primary px-4 py-2 text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
            >
              Get started
            </Link>
            <Link
              href="/docs/models"
              className="rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:bg-fd-accent"
            >
              Model zoo
            </Link>
            <a
              href={links.hfSpace}
              rel="noreferrer noopener"
              target="_blank"
              className="rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:bg-fd-accent"
            >
              Try the demo
            </a>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <DynamicCodeBlock lang="bash" code="uv add nobg" />
          <DynamicCodeBlock lang="python" code={quickStart} />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <figure className="flex flex-col gap-2 rounded-xl border bg-fd-card p-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="https://raw.githubusercontent.com/feyninc/nobg/main/assets/feyn_mark.png"
              alt="Input photo with its original background"
              className="rounded-lg"
            />
            <figcaption className="text-xs text-fd-muted-foreground">input.jpg</figcaption>
          </figure>
          <figure className="flex flex-col gap-2 rounded-xl border bg-fd-card p-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="https://raw.githubusercontent.com/feyninc/nobg/main/assets/feyn_mark_cutout.png"
              alt="The same photo with its background removed"
              className="rounded-lg"
            />
            <figcaption className="text-xs text-fd-muted-foreground">output.png</figcaption>
          </figure>
        </div>
      </section>

      <section className="mx-auto w-full max-w-5xl border-t px-4 py-16">
        <h2 className="text-2xl font-semibold tracking-tight">What you get</h2>
        <div className="mt-8 grid gap-px overflow-hidden rounded-xl border bg-fd-border sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <div key={feature.title} className="bg-fd-card p-5">
              <h3 className="text-sm font-semibold">{feature.title}</h3>
              <p className="mt-2 text-sm text-fd-muted-foreground">{feature.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto w-full max-w-5xl border-t px-4 py-16">
        <h2 className="text-2xl font-semibold tracking-tight">Start here</h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { href: '/docs/installation', title: 'Installation', body: 'uv, pip and the extras.' },
            {
              href: '/docs/quickstart',
              title: 'Quickstart',
              body: 'From a photo to a cutout in three lines.',
            },
            {
              href: '/docs/guides/text-prompts',
              title: 'Text prompts',
              body: 'Cut out a named subject with MultiMatte.',
            },
            {
              href: '/docs/reference/birefnet',
              title: 'API reference',
              body: 'Every class, method and config field.',
            },
          ].map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-xl border bg-fd-card p-4 transition-colors hover:bg-fd-accent"
            >
              <h3 className="text-sm font-semibold">{item.title}</h3>
              <p className="mt-1.5 text-sm text-fd-muted-foreground">{item.body}</p>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}

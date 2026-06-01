import type { NextConfig } from "next";

/**
 * Static-export configuration for offline desktop embedding.
 *
 * `output: 'export'` makes `next build` emit a fully static site into `out/`,
 * which the PySide6 launcher serves from a local loopback HTTP server (no Node
 * runtime, no internet). `images.unoptimized` is required because the Next image
 * optimizer needs a running server, which the offline embed does not have.
 *
 * The embed route lives at `/embed` -> `out/embed/index.html`.
 */
const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true,
  },
  // Static export emits a directory per route (out/embed/index.html); the
  // trailing slash keeps asset/route resolution clean under the loopback server.
  trailingSlash: true,
};

export default nextConfig;

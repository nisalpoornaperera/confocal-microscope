// Regenerates src/api/schema.ts from the backend's OpenAPI schema.
//
// The schema is produced offline by the backend itself (create_app() builds the
// FastAPI app without starting the hardware; that only happens in the lifespan),
// so no server needs to be running:
//
//   uv run --directory ../backend python -c "...create_app().openapi()..."
//
// then openapi-typescript turns it into TypeScript types. Requires uv on PATH.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const schemaJson = resolve(root, "openapi.json");
const output = resolve(root, "src", "api", "schema.ts");

const python =
  "import json; from confocal.api.app import create_app; print(json.dumps(create_app().openapi()))";
const json = execFileSync("uv", ["run", "--directory", "../backend", "python", "-c", python], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
  stdio: ["ignore", "pipe", "inherit"],
});
const schema = JSON.parse(json);
writeFileSync(schemaJson, JSON.stringify(schema, null, 2) + "\n");
mkdirSync(dirname(output), { recursive: true });

const cli = resolve(root, "node_modules", "openapi-typescript", "bin", "cli.js");
execFileSync(process.execPath, [cli, schemaJson, "--output", output], {
  cwd: root,
  stdio: "inherit",
});
console.log(`OpenAPI ${schema.info?.version ?? "?"} -> ${output}`);

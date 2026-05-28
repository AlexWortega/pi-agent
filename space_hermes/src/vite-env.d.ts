/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_LOG_API?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}

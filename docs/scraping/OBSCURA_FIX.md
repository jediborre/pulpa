> **Ubicación original:** scratch/OBSCURA_FIX.md

---

# Fix para Obscura: Cookies en fetch() cross-origin

## Problema

En `crates/obscura-js/src/ops.rs`, la función `op_fetch_url` (que implementa `fetch()` 
en el runtime JavaScript de V8) solo envía cookies cuando el request es **same-origin**:

```rust
// Línea ~350 en ops.rs (v0.1.5)
if !is_cross_origin {
    if let Some(ref jar) = cookie_jar {
        if let Ok(parsed_url) = url::Url::parse(&current_url) {
            let cookie_header = jar.get_cookie_header(&parsed_url);
            if !cookie_header.is_empty() {
                req = req.header("Cookie", &cookie_header);
            }
        }
    }
}
```

Esto causa que `fetch('https://api.sofascore.com/...')` desde una página en 
`https://www.sofascore.com/...` NO envíe cookies, porque los orígenes son diferentes 
(`www.sofascore.com` ≠ `api.sofascore.com`).

Un navegador real envía cookies basándose en el **dominio de la cookie** (ej: `.sofascore.com`), 
no en si el request es same-origin.

## Fix

Cambiar la condición `if !is_cross_origin` para que SIEMPRE envíe cookies basándose 
en el dominio de la cookie:

```rust
// ANTES (incorrecto):
if !is_cross_origin {
    if let Some(ref jar) = cookie_jar {
        if let Ok(parsed_url) = url::Url::parse(&current_url) {
            let cookie_header = jar.get_cookie_header(&parsed_url);
            if !cookie_header.is_empty() {
                req = req.header("Cookie", &cookie_header);
            }
        }
    }
}

// DESPUÉS (correcto - comportamiento de navegador real):
if let Some(ref jar) = cookie_jar {
    if let Ok(parsed_url) = url::Url::parse(&current_url) {
        let cookie_header = jar.get_cookie_header(&parsed_url);
        if !cookie_header.is_empty() {
            req = req.header("Cookie", &cookie_header);
        }
    }
}
```

## Archivo a modificar

`crates/obscura-js/src/ops.rs` - función `op_fetch_url`, dentro del loop de redirects.

## Cómo compilar

```bash
git clone https://github.com/h4ckf0r0day/obscura.git
cd obscura

# Aplicar el fix (editar ops.rs)
# ...

# Compilar
cargo build --release --features stealth

# El binario estará en target/release/obscura
```

## Alternativa sin compilar

Usar `obscura fetch` con `--eval` para ejecutar JavaScript sincrónico que extraiga 
datos directamente del DOM renderizado, sin necesidad de hacer `fetch()` cross-origin.

O usar el scraper actual con Chrome headless que ya funciona correctamente.

## Impacto

Este fix permitiría que Obscura funcione con APIs que están en subdominios diferentes 
al de la página principal (como `api.sofascore.com` desde `www.sofascore.com`), 
que es un patrón muy común en sitios web modernos.

"""Descarga las imagenes de prueba desde Wikimedia Commons y registra su procedencia.

Los archivos estan fijados por titulo (no por busqueda) para que la corrida sea reproducible:
cualquiera que ejecute el script obtiene exactamente las mismas tres imagenes.

Las tres condiciones varian la cantidad de contexto que rodea al objeto de la clase:
    close_up  -> el balon ocupa casi todo el encuadre
    campo     -> balon nitido sobre cesped, con estadio de fondo
    partido   -> escena completa de un partido, el balon es un objeto pequeno mas
"""

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "LoopEngineering/0.1 (proyecto academico UVG; explicabilidad SHAP)"
THUMB_WIDTH = 1024

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "images"

# id local -> titulo exacto en Commons
IMAGES = {
    "close_up": "File:Soccer Ball (42232038211).jpg",
    "campo": "File:Adidas soccer ball on a grass pitch (Unsplash).jpg",
    "partido": "File:Players and referees before a football match.jpg",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read()


def _plain(extmetadata: dict, key: str) -> str:
    """Extrae un campo de extmetadata quitando el HTML que a veces trae Commons."""
    raw = extmetadata.get(key, {}).get("value", "")
    text = re.sub(r"<[^>]+>", "", str(raw))
    return " ".join(text.split())


def fetch_metadata(titles: list[str]) -> dict:
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": str(THUMB_WIDTH),
        "format": "json",
        "formatversion": "2",
    }
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params)))
    return {page["title"]: page["imageinfo"][0] for page in data["query"]["pages"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = fetch_metadata(list(IMAGES.values()))

    manifest = []
    for local_id, title in IMAGES.items():
        item = info[title]
        meta = item.get("extmetadata", {})
        blob = _get(item["thumburl"])
        path = OUT_DIR / f"{local_id}.jpg"
        path.write_bytes(blob)

        manifest.append(
            {
                "id": local_id,
                "archivo": path.name,
                "titulo_commons": title,
                "pagina": item["descriptionurl"],
                "url_descarga": item["thumburl"],
                "autor": _plain(meta, "Artist"),
                "licencia": _plain(meta, "LicenseShortName"),
                "url_licencia": meta.get("LicenseUrl", {}).get("value", ""),
                "ancho_original": item["width"],
                "alto_original": item["height"],
                "sha256": hashlib.sha256(blob).hexdigest(),
                "bytes": len(blob),
            }
        )
        print(f"{local_id:9s} {len(blob):>8,d} B  {manifest[-1]['licencia']}  {title}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {"fuente": "Wikimedia Commons", "ancho_solicitado": THUMB_WIDTH, "imagenes": manifest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nManifiesto -> {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

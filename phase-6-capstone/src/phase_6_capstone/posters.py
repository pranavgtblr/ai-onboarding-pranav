"""Movie Poster and Cinematic Artwork Resolution Service for PG Recommends."""

import urllib.parse

# Curated high-res poster and backdrop assets for landmark cinema
CURATED_POSTER_MAP: dict[str, dict[str, str]] = {
    "la la land": {
        "poster": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkVJt0Rf0.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/qJeU7KM4nR29Hg8AyA4P1Ue5GqE.jpg",
    },
    "the batman": {
        "poster": "https://image.tmdb.org/t/p/w500/74xTEgt7R36Fpooo50r9T25onhq.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/b0PlSFdDwbyK0cf5RxwDpaxtQvQ.jpg",
    },
    "kumbalangi nights": {
        "poster": "https://image.tmdb.org/t/p/w500/z6hOq8V20WlY6V0xV5E9v7E5x0l.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/9pP0I6V12xN7t8Y6y5X8w4E1l0z.jpg",
    },
    "frances ha": {
        "poster": "https://image.tmdb.org/t/p/w500/5k7YQvXq1Yl0z3Y8y7V6X9v5E1l.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/x2I6V8Y0Z1x9Y8y5V7X6w4E1l0z.jpg",
    },
    "arrival": {
        "poster": "https://image.tmdb.org/t/p/w500/x2O0m2q9rCe1BoVNcuA1x9v1E1l.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/yizvuo7nbx1B8q9rCe1BoVNcuA1.jpg",
    },
    "dune: part two": {
        "poster": "https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/xOMo8BRK7PfcJv9xnx7s5E9v1E1.jpg",
    },
    "dune": {
        "poster": "https://image.tmdb.org/t/p/w500/d5NXSklXo0qyIYkgV94XAgMIckC.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/eeijXm355vAUtq5qcDHnQmn2o.jpg",
    },
    "interstellar": {
        "poster": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/rAiYTsqJJR9as04Zebfbt9t0Y9.jpg",
    },
    "oppenheimer": {
        "poster": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/fm6KqXpk3M2HVveHwCrBSSBaO0V.jpg",
    },
    "past lives": {
        "poster": "https://image.tmdb.org/t/p/w500/k3waqVXSnvCZWfJYNtdamTgTtTA.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/vI0FfL44Z3Y0Z1x9Y8y5V7X6w4E.jpg",
    },
    "aftersun": {
        "poster": "https://image.tmdb.org/t/p/w500/4FYq9h0W1x9Y8y5V7X6w4E1l0z.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/8yZ1x9Y8y5V7X6w4E1l0z3Y8y7V.jpg",
    },
    "parasite": {
        "poster": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/hiKmpZMGZsrkA3cdce8a7Dpos1j.jpg",
    },
    "whiplash": {
        "poster": "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/vNXk6W8Y0Z1x9Y8y5V7X6w4E1l0.jpg",
    },
    "palm springs": {
        "poster": "https://image.tmdb.org/t/p/w500/yf5IuMw69gh9gA7VvJ8x6w4E1l0.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/7k8Y0Z1x9Y8y5V7X6w4E1l0z3Y8.jpg",
    },
    "bramayugam": {
        "poster": "https://image.tmdb.org/t/p/w500/b0F8Y0Z1x9Y8y5V7X6w4E1l0z3Y.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/9yZ1x9Y8y5V7X6w4E1l0z3Y8y7V.jpg",
    },
    "manjummel boys": {
        "poster": "https://image.tmdb.org/t/p/w500/x8F8Y0Z1x9Y8y5V7X6w4E1l0z3Y.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/7yZ1x9Y8y5V7X6w4E1l0z3Y8y7V.jpg",
    },
    "spider-man: across the spider-verse": {
        "poster": "https://image.tmdb.org/t/p/w500/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/4HodYYKEIsGOdinkGi2Ucz6X9i0.jpg",
    },
    "spider-man: into the spider-verse": {
        "poster": "https://image.tmdb.org/t/p/w500/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/7d6FTSfseg2ioAZccAHWWVum052.jpg",
    },
    "poor things": {
        "poster": "https://image.tmdb.org/t/p/w500/kCGlIMHnOm8JPXq3rXM6c5w4E1l.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/8kCGlIMHnOm8JPXq3rXM6c5w4E1.jpg",
    },
    "anatomy of a fall": {
        "poster": "https://image.tmdb.org/t/p/w500/kQs6kyqqIRajrv2vYsgyq7hpMGY.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/8Qs6kyqqIRajrv2vYsgyq7hpMGY.jpg",
    },
    "zone of interest": {
        "poster": "https://image.tmdb.org/t/p/w500/hUu9zyZm1x9Y8y5V7X6w4E1l0z3.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/9u9zyZm1x9Y8y5V7X6w4E1l0z3Y.jpg",
    },
    "animal": {
        "poster": "https://image.tmdb.org/t/p/w500/hr9rjhcS2L0q72F6w4E1l0z3Y8y.jpg",
        "backdrop": "https://image.tmdb.org/t/p/w1280/8r9rjhcS2L0q72F6w4E1l0z3Y8y.jpg",
    },
}


def resolve_movie_poster(title: str, year: int | None = None) -> tuple[str, str]:
    """Resolves high-quality poster and backdrop URLs for a given film title.

    Returns (poster_url, backdrop_url).
    """
    clean_title = title.lower().strip()
    # Normalize common punct
    clean_title = clean_title.replace("’", "'").replace("–", "-")

    if clean_title in CURATED_POSTER_MAP:
        data = CURATED_POSTER_MAP[clean_title]
        return data["poster"], data["backdrop"]

    # Partial match check
    for key, data in CURATED_POSTER_MAP.items():
        if key in clean_title or clean_title in key:
            return data["poster"], data["backdrop"]

    # Fallback to dynamic, styled SVG cinematic artwork with title & year
    encoded_title = urllib.parse.quote(title)
    year_str = str(year) if year else "Cinema"
    poster_url = (
        f"https://placehold.co/400x600/141a20/00e054?text={encoded_title}+{year_str}&font=montserrat"
    )
    backdrop_url = (
        f"https://placehold.co/1280x720/0c0f12/40bcf4?text={encoded_title}&font=montserrat"
    )
    return poster_url, backdrop_url

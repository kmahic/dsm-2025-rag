import time
from typing import List
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine


RANKING_LOCATION = "eu"
RANKING_ENDPOINT = "eu-discoveryengine.googleapis.com"
MAX_CANDIDATES = 200
MAX_TITLE_LEN = 1024
MAX_CONTENT_LEN = 20_000

_rank_client = discoveryengine.RankServiceClient(
    client_options=ClientOptions(api_endpoint=RANKING_ENDPOINT)
)


def _ranking_config_path(project_id: str, location: str = RANKING_LOCATION) -> str:
    return _rank_client.ranking_config_path(
        project=project_id,
        location=location,
        ranking_config="default_ranking_config",
    )


def rerank_documents(
    project_id: str,
    query: str,
    docs: List[dict],
    *,
    model: str = "semantic-ranker-default@latest",
    top_n: int | None = None,
) -> tuple[List[dict], float]:
    if not docs:
        return docs, 0.0

    candidates = docs[:MAX_CANDIDATES]
    records = [
        discoveryengine.RankingRecord(
            id=str(d.get("id", idx)),
            title=(d.get("title", ""))[:MAX_TITLE_LEN] or None,
            content=(d.get("content", ""))[:MAX_CONTENT_LEN],
        )
        for idx, d in enumerate(candidates)
    ]

    req = discoveryengine.RankRequest(
        ranking_config=_ranking_config_path(project_id),
        query=query,
        records=records,
        model=model,
        top_n=top_n or len(records),
    )

    start = time.time()
    resp = _rank_client.rank(request=req)
    elapsed = time.time() - start

    order = {r.id: (idx, r.score) for idx, r in enumerate(resp.records)}
    ranked = sorted(
        (d for d in candidates if str(d.get("id")) in order),
        key=lambda d: order[str(d.get("id"))][0]
    )

    for d in ranked:
        d["rerank_score"] = order[str(d.get("id"))][1]

    return ranked, elapsed

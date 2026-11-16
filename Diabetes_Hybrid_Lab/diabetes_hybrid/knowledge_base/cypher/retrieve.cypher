MATCH (anchor:Concept)
WHERE anchor.id IN $anchors
CALL (anchor) {
  MATCH (p:Passage)-[:ABOUT]->(anchor)
  RETURN p, anchor.id AS matched, 0 AS distance
  UNION
  MATCH (anchor)-[rel]-(neighbor:Concept)<-[:ABOUT]-(p:Passage)
  WHERE rel.passage_id = p.id
  RETURN p, anchor.id AS matched, 1 AS distance
}
MATCH (p)-[:FROM_SOURCE]->(source:Source)
WITH p, source, collect(DISTINCT {id: matched, distance: distance}) AS paths
OPTIONAL MATCH (left:Concept)-[r]->(right:Concept)
WHERE r.passage_id = p.id
RETURN p.id AS id,
       CASE WHEN $language = 'vi' THEN p.title_vi ELSE p.title END AS title,
       CASE WHEN $language = 'vi' THEN p.text_vi ELSE p.text END AS text,
       p.about AS about, p.focus AS focus, coalesce(p.claims, []) AS claims, paths,
       source {.publisher, .url, .kind, .tier,
               title: CASE WHEN $language = 'vi' THEN source.title_vi ELSE source.title END} AS source,
       collect(CASE WHEN r IS NOT NULL THEN
         {from: CASE WHEN $language = 'vi' THEN left.name_vi ELSE left.name END,
          relation: type(r), to: CASE WHEN $language = 'vi' THEN right.name_vi ELSE right.name END}
       END) AS edges
ORDER BY id

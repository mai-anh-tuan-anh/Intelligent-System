// 1. Graph visualization in the local Neo4j Browser
MATCH path = (:Passage)-[:ABOUT]->(:Concept)
RETURN path LIMIT 50;

// 2. A type-1 relationship and the passage and source that support it
MATCH path = (:Concept)<-[r:CLASSIFIED_WITH]-(d:Concept {id:'type1'})
             <-[:ABOUT]-(p:Passage)-[:FROM_SOURCE]->(s:Source)
WHERE p.id = r.passage_id
RETURN path, p.text, s.url;

// 3. Two hops from risk factors to prevention of type 2
MATCH path = (factor:Concept)<-[:HAS_RISK_FACTOR]-(:Concept {id:'type2'})
             -[:HAS_PREVENTION_TOPIC]->(prevention:Concept)
RETURN path;

// 4. PDF provenance
MATCH path = (:Concept {id:'survey'})<-[:ABOUT]-(p:Passage)-[:FROM_SOURCE]->(s:Source {kind:'PDF'})
RETURN path, p.text, s.url;

// 5. Trace the Vinmec result from the demo back to its source
MATCH path = (c:Concept {id:'type1'})<-[:ABOUT]-(p:Passage {id:'p02'})
             -[:FROM_SOURCE]->(s:Source {id:'vinmec'})
RETURN path, p.id AS passage_id, p.text_vi AS passage, s.url AS source_url;

// 6. Compare source bundles after the merge
MATCH (p:Passage)-[:FROM_SOURCE]->(s:Source)
RETURN s.publisher AS publisher, s.kind AS kind, count(DISTINCT s) AS sources,
       count(DISTINCT p) AS passages
ORDER BY sources DESC, publisher;

"""Apply the canonical Neo4j seed, with statement results checked."""
import os

from knowledge_base.build_kb import build, cypher_statements
from core.rag import GraphRAG, connect


def main():
    data = build(check=True)
    with connect() as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            for statement in cypher_statements(data):
                session.run(statement).consume()
        print(GraphRAG(driver).status())


if __name__ == "__main__":
    main()


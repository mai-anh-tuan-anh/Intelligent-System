# Diabetes Hybrid Lab — Presentation Questions and Answers

## A. Deep learning

### Q1. Which model did you use, and why?

**Answer.** The live demo defaults to the original Assignment 4 PyTorch MLP and fitted scaler. The model has widths 21 → 64 → 32 → 3 and ReLU hidden layers. We reused its trained weights. The selectable grouped-evaluation model adds dropout. BRFSS presents a fixed-length tabular vector, which suits an MLP. The two supplementary binary tasks use separate models with 32- and 16-unit hidden layers.

### Q2. What are the datasets, inputs, and targets?

**Answer.** The primary BRFSS 2015 table has 253,680 rows and 21 coded survey features. It predicts no reported diabetes, reported prediabetes, or reported diabetes. The 100,000-row Assignment 2 benchmark predicts its supplied binary label with ten selected features. The UCI benchmark has 101,766 historical encounters and predicts a recorded readmission within 30 days. The UCI task uses group splits by patient. The English notebook lists actual test sizes, feature names, and source links.

### Q3. What are the experimental results?

**Answer.** The original model reported 0.4252 macro-F1 on the Assignment 4 split. In the separate grouped experiment, MLP macro-F1 was 0.4237, versus 0.4249 for logistic regression on the same 37,938 test rows. Prediabetes recall was 0.3253 for logistic regression and 0.2395 for the grouped MLP. The majority baseline illustrates the imbalance with 0.8411 accuracy and zero prediabetes recall. Independent MLP benchmarks reached macro-F1 0.6056 on Assignment 2 and 0.5227 on UCI. [BRFSS metrics](../artifacts/experiment.json) · [Supplementary metrics](../artifacts/additional_experiments.json)

### Q4. How did you use earlier assignments?

**Answer.** Assignment 4 supplied the processed BRFSS dataset and the archived MNIST/CIFAR-10 CNN benchmarks. Assignment 2 supplied an additional tabular benchmark. The earlier project supplied the Pima comparator and the design for modular routing. For the oral comparison, Assignment 3 predicts a continuous HDB resale price, while Assignment 6 predicts AAPL prices from an ordered series. Their RMSE values have different units and belong to distinct test protocols.

## B. Knowledge graph and RAG

### Q5. Where is the Neo4j knowledge base?

**Answer.** The local seed is [kb_diabet.cypher](../knowledge_base/cypher/kb_diabet.cypher). [build_kb.py](../knowledge_base/build_kb.py) combines individual URL bundles into [kb_diabetes.json](../knowledge_base/kb_diabetes.json) and generates the seed. Open [Neo4j Browser](http://127.0.0.1:7476/browser/), select the `neo4j` database, and run [example Cypher queries](../knowledge_base/cypher/examples.cypher). The three node types are Concept, Passage, and Source. A passage links its topic and publisher. Semantic edges identify the supporting passage.

### Q6. Which websites and PDFs are included?

**Answer.** WHO, CDC, and NIDDK supply the core public-health and patient-education material. Vinmec and FPT Long Chau supplement Vietnamese-language coverage. The original CDC survey codebook is linked directly to the relevant PDF page. The seed stores reviewed paraphrases, source URLs, and source categories. [Source catalogue](../README.md#where-are-the-knowledge-graph-and-retrieval-code)

### Q7. What code implements the RAG-KB?

**Answer.** [rag.py](../core/rag.py) matches Vietnamese and English aliases and ranks Neo4j evidence. [retrieve.cypher](../knowledge_base/cypher/retrieve.cypher) performs the parameterized graph traversal. Character TF-IDF measures lexical similarity. Graph edges preserve semantic structure and traceable evidence. The application uses extractive answers. There is no generative LLM in this version.

### Q8. How does the assistant handle a follow-up?

**Answer.** After a successful answer about one diabetes type, the client retains that concept ID in memory. A follow-up explicitly continuing the topic can reuse that ID. A new explicit type takes precedence. If the referenced topic cannot be inferred reliably, the assistant asks for clarification or reports insufficient evidence. Clearing the conversation removes local context.

## C. Example assistant questions

### Q9. What is gestational diabetes, and when is screening usually done?

**Answer.** Gestational diabetes is a form of diabetes that develops during pregnancy. Prenatal screening can identify it when symptoms are absent. NIDDK describes usual testing at weeks 24–28, with earlier evaluation for selected patients. The prenatal care team chooses and interprets the test. [NIDDK overview](https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/gestational) · [NIDDK testing](https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/gestational/tests-diagnosis)

### Q10. What does HbA1c measure, and does it require fasting?

**Answer.** HbA1c reflects average glucose over roughly three months and usually requires no fasting. A fasting plasma glucose test uses different preparation. Confirm the exact test with the testing site. [NIDDK HbA1c](https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test) · [FPT Long Chau](https://nhathuoclongchau.com.vn/bai-viet/truoc-khi-xet-nghiem-tieu-duong-co-can-nhin-an-hay-khong.html)

### Q11. What may reduce the chance of developing type 2 diabetes?

**Answer.** WHO describes regular physical activity, healthy eating, maintaining a healthy weight, and avoiding tobacco as measures that may help prevent or delay type 2 diabetes. [WHO diabetes fact sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes)

### Q12. Can the assistant tell a person whether they have diabetes or how much medication to take?

**Answer.** It explains general testing concepts and routes a personal result to a healthcare professional. It can suggest what to bring to the visit, such as the test report, sampling time, symptom history, and medication list. Medication decisions are routed to the prescriber or pharmacist. The BRFSS output is kept separate from these replies.

### Q13. What happens if a question mentions severe symptoms?

**Answer.** Explicit red-flag wording such as loss of consciousness or difficulty breathing triggers an immediate emergency-care message, even when Neo4j is unavailable. The rule is a limited routing feature and has no validated triage sensitivity. [CDC diabetic ketoacidosis](https://www.cdc.gov/diabetes/about/diabetic-ketoacidosis.html)

## D. Architecture, demo, and evidence

### Q14. What does each agent do?

**Answer.** Coordinator selects the requested branch. Data Agent validates the synthetic survey input. Prediction Agent applies the trained PyTorch model. Rule Agent checks both question scope and score separation. Knowledge Agent retrieves graph evidence. Explanation Agent combines these results into a report with separate predictions, rules, evidence, review status, and limitations. Monitoring publishes a privacy-filtered trace with status, evidence identifiers, version, and latency.

### Q15. Which demonstrations should be shown?

**Answer.** First run the English [Jupyter Notebook](../notebooks/Diabetes_Hybrid_Lab.ipynb) to show the three benchmark results, model inference, and live Neo4j queries. In the [Vietnamese browser demo](http://127.0.0.1:8011/), ask a question about gestational diabetes, then ask about tests without repeating the type. Open the evidence links. Next choose a synthetic profile and run both branches with a type 1 question. Finally show a personal-result or emergency-routing example and the corresponding trace.

### Q16. How was it evaluated, and what would clinical deployment require?

**Answer.** The experiments use held-out splits. The assistant tests cover retrieval, provenance, bilingual retrieval, follow-up context, unsupported questions, medication requests, and emergency routing. Tests are drawn from the curated knowledge coverage. Clinical deployment would require a defined indication, representative patient-level datasets, external and prospective validation, oversight by clinical professionals, and a validated process for monitoring errors. [Verification results](../artifacts/demo_verification.json)

### Q17. What happens when the result is uncertain or the sources disagree?

**Answer.** A margin below 0.10 between the two highest model scores adds a review reason. If approved retrieved sources disagree on the same curated scope key, the assistant shows the source links, abstains from choosing a claim, and recommends professional review. The conflict test uses synthetic evidence, and the detector only covers annotated assertions in retrieved passages.


# P10 — Évaluez les performances d'un LLM

## Assistant NBA hybride RAG + SQL

Ce projet a été réalisé dans le cadre du projet OpenClassrooms **« Évaluez les performances d'un LLM »**.

L'objectif est d'auditer puis d'améliorer un prototype d'assistant IA destiné à l'analyse de données NBA.

Le prototype initial reposait principalement sur une architecture RAG (*Retrieval-Augmented Generation*) appliquée à des documents textuels.

Le système a ensuite été enrichi avec :

- une validation structurée avec Pydantic ;
- une validation sémantique avec Pydantic AI ;
- une observabilité avec Logfire ;
- une évaluation automatisée avec RAGAS ;
- une base de données SQLite ;
- un Tool SQL LangChain ;
- un agent hybride capable de choisir entre RAG et SQL ;
- des benchmarks avant / après ;
- des graphiques comparatifs.

---

## Objectifs du projet

Le projet répond à trois objectifs principaux.

### 1. Évaluer le système RAG initial

L'objectif est de mesurer objectivement les performances du prototype sur plusieurs catégories de questions métier :

- simples ;
- complexes ;
- bruitées.

Les métriques RAGAS utilisées sont :

- Context Precision ;
- Context Recall ;
- Faithfulness ;
- Answer Relevancy.

### 2. Ajouter une couche SQL

Les données numériques provenant du fichier Excel sont chargées dans une base SQLite.

Un Tool SQL LangChain permet ensuite de transformer dynamiquement une question en langage naturel en requête SQL.

### 3. Comparer les performances avant et après enrichissement

Le système final distingue :

- les questions documentaires, traitées avec le RAG ;
- les questions quantitatives, traitées avec SQL.

---

# Architecture

```text
                          Question utilisateur
                                  |
                                  v
                         +------------------+
                         |   Agent hybride  |
                         +------------------+
                            /            \
                           /              \
                          v                v
                  +-------------+    +-------------+
                  |     RAG     |    |     SQL     |
                  +-------------+    +-------------+
                        |                  |
                        v                  v
                  Embeddings          NL -> SQL
                        |                  |
                        v                  v
                     FAISS             SQLite
                        |                  |
                        v                  v
                  Top-k chunks       Résultats SQL
                        |                  |
                        +--------+---------+
                                 |
                                 v
                         Réponse structurée
```

---

# Pipeline RAG

La branche RAG traite les informations non structurées issues principalement des PDF Reddit.

Le pipeline comprend :

1. chargement des documents ;
2. nettoyage ;
3. découpage en chunks ;
4. validation Pydantic ;
5. contrôle sémantique Pydantic AI ;
6. génération des embeddings ;
7. indexation FAISS ;
8. retrieval des chunks pertinents ;
9. génération de la réponse ;
10. validation structurée de la sortie.

Le modèle d'embedding utilisé est :

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Le vector store utilise :

```text
FAISS
IndexFlatIP
Embeddings normalisés
```

L'index utilisé pendant les évaluations contient :

```text
302 vecteurs
302 chunks
```

---

# Validation avec Pydantic

Pydantic est utilisé pour sécuriser les structures manipulées par le pipeline.

Les validations couvrent notamment :

- les chunks préparés ;
- les identifiants ;
- les embeddings ;
- les dimensions des vecteurs ;
- les valeurs numériques non finies ;
- les réponses structurées générées.

Exemples de modèles :

```text
PreparedChunk
EmbeddingBatch
RAGAnswer
```

---

# Validation sémantique avec Pydantic AI

Une seconde couche permet d'évaluer si un chunk est exploitable pour le système RAG.

Le validateur retourne :

```text
is_valid
relevance_score
readability_score
reason
```

Les scores sont compris entre :

```text
0.0 et 1.0
```

Exemple réel observé :

```text
Chunk exploitable

is_valid = True
relevance_score = 0.95
readability_score = 0.92
```

Pour un fragment composé essentiellement de navigation et d'URLs :

```text
is_valid = False
relevance_score = 0.00
readability_score = 0.00
```

---

# Observabilité

Pydantic Logfire est intégré afin de tracer pas à pas le pipeline RAG / SQL / HYBRID ainsi que les appels Pydantic AI.

L'observabilité permet notamment d'inspecter :

- les appels au modèle ;
- les validations ;
- les erreurs de structure ;
- les étapes de génération ;
- les temps d'exécution.

La configuration utilise `send_to_logfire=False` : les traces du routage, de l'exécution SQL, du retrieval RAG et de la synthèse hybride sont visualisées localement dans la console, sans envoi vers Logfire Cloud.

---

# Base de données

Les données numériques du fichier Excel sont intégrées dans une base SQLite.

Le schéma relationnel contient :

```text
players
matches
stats
reports
```

Le schéma SQL est disponible dans :

```text
db/schema.sql
```

La base utilisée localement est :

```text
data/nba_rag.db
```

Le pipeline d'ingestion est :

```text
load_excel_to_db.py
```

Les données sont validées avec Pydantic avant insertion.

## Limitation importante

Le fichier Excel fourni contient principalement des statistiques agrégées.

Il ne contient pas un historique complet match par match.

La table `matches` n'est donc pas artificiellement remplie avec des informations qui n'existent pas dans les données sources.

---

# Tool SQL LangChain

Le Tool SQL permet de transformer une question en langage naturel en requête SQL.

Il effectue notamment :

```text
Question utilisateur
        |
        v
Inspection du schéma
        |
        v
Prompt + few-shot
        |
        v
Génération SQL
        |
        v
Validation read-only
        |
        v
Exécution SQLite
        |
        v
Résultat
```

Les requêtes autorisées doivent rester en lecture seule.

Les principales formes attendues sont :

```sql
SELECT ...
```

et :

```sql
WITH ...
SELECT ...
```

Les opérations de modification sont refusées.

---

# Agent hybride

L'agent final choisit entre les deux sources d'information.

## Exemple RAG

```text
Selon les discussions Reddit, quelle faiblesse
de Reggie Miller est mentionnée lorsqu'il doit
créer son propre tir ?
```

Cette question est traitée à partir du corpus documentaire.

## Exemple SQL

```text
Quelle équipe a marqué le plus de points ?
```

Cette question est envoyée vers le Tool SQL.

Une détection déterministe est utilisée pour certaines intentions numériques.

Elle prend notamment en compte des termes comme :

```text
points
rebonds
passes
assists
pourcentage
moyenne
total
classement
top N
```

Des exclusions sont également prévues pour certaines formulations négatives.

---

# Installation

## Prérequis

Environnement utilisé pendant le développement :

```text
Python 3.11.9
Ollama 0.34.2
qwen2.5:7b-instruct
```

Le projet utilise également notamment :

```text
ragas 0.3.9
langchain 0.3.23
langchain-core 0.3.86
langchain-ollama 0.2.3
pydantic 2.x
pydantic-ai
FAISS
SentenceTransformers
SQLite
Pytest
Matplotlib
```

---

## 1. Cloner le repository

```bash
git clone https://github.com/VincentDesmouceaux/P10-LLM-RAG-Evaluation.git
cd P10-LLM-RAG-Evaluation
```

---

## 2. Créer l'environnement virtuel

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Vérification :

```bash
python -m pip check
```

---

# Installation d'Ollama

Sur macOS avec Homebrew :

```bash
brew install ollama
brew services start ollama
```

Vérifier :

```bash
ollama --version
```

Télécharger le modèle local :

```bash
ollama pull qwen2.5:7b-instruct
```

Vérifier sa présence :

```bash
ollama list
```

Test rapide :

```bash
ollama run qwen2.5:7b-instruct "Réponds uniquement par OK."
```

Résultat attendu :

```text
OK
```

---

# Données

Les fichiers sources sont placés dans :

```text
inputs/
```

Le projet utilise notamment :

```text
inputs/
├── Reddit 1.pdf
├── Reddit 2.pdf
├── Reddit 3.pdf
├── Reddit 4.pdf
└── regular NBA.xlsx
```

---

# Indexation RAG

Pour reconstruire l'index documentaire :

```bash
python -m scripts.indexer
```

Le vector store est enregistré dans :

```text
vector_db/
```

Il contient notamment :

```text
faiss_index.idx
document_chunks.pkl
```

---

# Chargement des données SQL

Pour reconstruire la base SQLite à partir du fichier Excel :

```bash
python load_excel_to_db.py
```

La base est créée dans :

```text
data/nba_rag.db
```

---

# Lancer les tests

```bash
python -m pytest -q
```

État observé lors de la dernière validation :

```text
11 passed
```

---

# Évaluation RAGAS

Le script principal d'évaluation est :

```text
scripts/evaluate_ragas.py
```

Exécution :

```bash
python -m scripts.evaluate_ragas
```

Une évaluation complémentaire du corpus Reddit est disponible avec :

```bash
python evaluate_reddit_ragas.py
```

Les résultats sont enregistrés dans :

```text
evaluation_results/
```

---

# Benchmark hybride

Pour comparer le comportement numérique avant et après intégration du SQL Tool :

```bash
python -m scripts.compare_before_after
```

Le CSV produit est :

```text
evaluation_results/before_after_hybrid_comparison.csv
```

---

# Résultats numériques

Le benchmark final comporte six scénarios fonctionnels.

## Avant SQL

```text
RAG seul
3 / 6 réponses correctes
Accuracy = 50 %
```

## Après SQL

```text
RAG + SQL Tool
6 / 6 réponses correctes
Accuracy = 100 %
```

L'amélioration observée sur ce benchmark est donc :

```text
+50 points de pourcentage
```

Ces résultats concernent uniquement les six scénarios fonctionnels testés et ne doivent pas être interprétés comme une précision générale de 100 % du système.

---

# Exemples numériques

Plusieurs résultats ont été correctement retrouvés par la branche SQL :

```text
Shai Gilgeous-Alexander
2485 points
```

```text
Oklahoma City
18 joueurs
```

```text
Detroit Pistons
10292 points
```

Comparaison filtrée :

```text
Oklahoma City : 9880
Brooklyn Nets : 7999
Différence    : 1881
```

---

# Résultats RAGAS

## Première évaluation

| Métrique | Score |
|---|---:|
| Context Precision | 0.8375 |
| Context Recall | 1.0000 |
| Faithfulness | 0.6071 |
| Answer Relevancy | 0.2804 |

## Après amélioration

| Métrique | Score |
|---|---:|
| Context Precision | 0.876 |
| Context Recall | 0.833 |
| Faithfulness | 0.905 |
| Answer Relevancy | 0.556 |

Évolution principale :

```text
Faithfulness
0.607 -> 0.905
```

```text
Answer Relevancy
0.280 -> 0.556
```

Le benchmark final comprend seulement trois questions textuelles :

```text
simple
complexe
bruitée
```

Les résultats doivent donc être interprétés comme une comparaison expérimentale du prototype, et non comme une validation statistique à grande échelle.

---

# Graphiques

Les graphiques sont disponibles dans :

```text
evaluation_results/figures/
```

Fichiers principaux :

```text
before_after_accuracy.png
before_after_by_category.png
ragas_before_after.png
sql_before_after.png
```

---

# Résultats CSV

Les fichiers principaux sont :

```text
evaluation_results/
├── before_after_hybrid_comparison.csv
├── final_evaluation_summary.csv
├── reddit_ragas_final.csv
└── retrieval_comparison.csv
```

---

# Rapport d'évaluation

L'analyse détaillée du projet est disponible dans :

```text
docs/evaluation_report.md
```

Le rapport présente notamment :

- la méthodologie ;
- l'architecture ;
- les résultats avant / après ;
- l'analyse RAGAS ;
- les limites du retrieval ;
- les limites du mapping NL vers SQL ;
- les biais des few-shot ;
- les limites du routage ;
- les limites du dataset ;
- les perspectives d'amélioration.

---

# Structure du projet

```text
P10_DSML/
├── data/
│   └── nba_rag.db
│
├── db/
│   └── schema.sql
│
├── docs/
│   └── evaluation_report.md
│
├── evaluation_results/
│   ├── figures/
│   ├── before_after_hybrid_comparison.csv
│   ├── final_evaluation_summary.csv
│   ├── reddit_ragas_final.csv
│   └── retrieval_comparison.csv
│
├── inputs/
│   ├── Reddit 1.pdf
│   ├── Reddit 2.pdf
│   ├── Reddit 3.pdf
│   ├── Reddit 4.pdf
│   └── regular NBA.xlsx
│
├── tests/
│   └── test_chunk_quality_validator.py
│
├── utils/
│   ├── chunk_quality_validator.py
│   ├── observability.py
│   ├── schemas.py
│   ├── structured_answer.py
│   └── vector_store.py
│
├── evaluate_reddit_ragas.py
├── hybrid_agent.py
├── scripts/
│   ├── __init__.py
│   ├── compare_before_after.py
│   ├── evaluate_ragas.py
│   └── indexer.py
├── load_excel_to_db.py
├── sql_tool.py
├── requirements.txt
└── README.md
```

Cette structure est volontairement simplifiée afin de mettre en évidence les composants principaux.

---

# Limites connues

Le système reste un prototype expérimental.

Les principales limites identifiées concernent :

- la petite taille du benchmark ;
- le bruit présent dans les PDF Reddit ;
- le caractère heuristique du routage ;
- les erreurs possibles de génération NL vers SQL ;
- l'influence des exemples few-shot ;
- la latence introduite par les réparations SQL ;
- l'absence de données détaillées match par match ;
- la dépendance des scores RAGAS au modèle juge ;
- la nécessité d'une validation qualitative en complément des métriques automatiques.

---

# Perspectives

Plusieurs améliorations peuvent être envisagées :

- augmenter le nombre de cas de test ;
- ajouter un reranker ;
- renforcer le classifieur de routage ;
- améliorer le nettoyage OCR ;
- ajouter davantage de tests SQL ;
- comparer plusieurs modèles juges ;
- ajouter des tests de régression ;
- intégrer un PlotTool pour produire dynamiquement des graphiques ;
- comparer EasyOCR et Nanonets OCR.

---

# Partie facultative

La partie facultative du projet propose notamment deux évolutions :

## OCR

Comparer :

```text
EasyOCR
```

avec :

```text
Nanonets OCR
```

afin de mesurer l'impact sur la qualité du texte extrait et sur les performances globales du RAG.

## PlotTool

Une fonctionnalité optionnelle de visualisation dynamique a été implémentée avec LangChain et Matplotlib.

Le fichier `plot_tool.py` génère automatiquement des graphiques à partir des données structurées retournées par SQLite.

### Pipeline

```text
Question utilisateur
        ↓
HybridNBAAgent
        ↓
Détection de l'intention graphique
        ↓
SQL Tool
        ↓
SQLite
        ↓
Données structurées
        ↓
PlotTool
        ↓
Matplotlib
        ↓
PNG dans generated_plots/
        ↓
Affichage Streamlit
```

Les routes principales `sql`, `rag` et `hybrid` sont conservées. La visualisation est ajoutée comme fonctionnalité complémentaire lorsqu'une demande graphique est détectée.

### Types de graphiques

Le Tool supporte `bar`, `line` et `pie`.

Les valeurs représentées proviennent des résultats SQL. Le LLM ne fabrique pas les données numériques du graphique.

### Exemple

`Montre-moi les 5 joueurs ayant marqué le plus de points sous forme de graphique.`

Le système détecte l'intention numérique et graphique, interroge SQLite, transmet les lignes retournées à `PlotTool`, génère un PNG puis l'affiche automatiquement dans Streamlit.

La génération est observable dans les traces Logfire avec le span `plot_generation`.

### Tests

La fonctionnalité est couverte par `tests/test_plot_tool.py` et `tests/test_hybrid_plot_integration.py`.

Les PNG générés sont exclus du versioning Git via `.gitignore`. Le fichier `generated_plots/.gitkeep` conserve le dossier dans l'architecture du projet.

Cette fonctionnalité est optionnelle et ne modifie pas le périmètre obligatoire figé dans la version `v1.0.0`.

---

# Auteur

Projet réalisé par **Vincent Desmouceaux** dans le cadre du parcours Data Scientist / Machine Learning OpenClassrooms.
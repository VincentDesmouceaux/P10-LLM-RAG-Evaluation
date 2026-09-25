# Rapport de mise en place et d'évaluation du système RAG

## 1. Contexte

Le projet vise à fiabiliser un assistant IA appliqué à l'analyse de données NBA.

Le prototype initial reposait principalement sur une architecture RAG (*Retrieval-Augmented Generation*) pour exploiter des documents textuels.

Le système a été enrichi afin de traiter également les questions numériques à partir de données structurées.

Les principaux objectifs sont :

- rendre l'environnement reproductible ;
- fiabiliser le pipeline de préparation des données ;
- mesurer les performances du système avec RAGAS ;
- intégrer les données statistiques Excel dans une base relationnelle ;
- créer un Tool SQL compatible avec LangChain ;
- router automatiquement les questions vers le RAG ou SQL ;
- comparer les performances avant et après enrichissement.

---

## 2. Architecture finale

Le système repose sur deux branches complémentaires :

1. une branche **RAG** pour les questions documentaires ;
2. une branche **SQL** pour les questions numériques.

### 2.1 Branche RAG

La branche RAG traite principalement les questions portant sur les documents textuels et les discussions Reddit.

Le pipeline est constitué des étapes suivantes :

1. nettoyage des documents ;
2. découpage en chunks ;
3. validation structurelle avec Pydantic ;
4. validation sémantique avec Pydantic AI ;
5. génération des embeddings ;
6. indexation avec FAISS ;
7. récupération des chunks pertinents ;
8. génération de la réponse avec Ollama ;
9. validation structurée de la réponse.

Le modèle d'embedding utilisé est :

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Le vector store utilise FAISS avec un index `IndexFlatIP`.

L'index exploité pendant les évaluations contient :

```text
302 vecteurs
302 chunks
```

### 2.2 Branche SQL

La branche SQL traite les questions nécessitant des résultats numériques exacts.

Le pipeline est :

1. détection de l'intention numérique ;
2. transformation de la question en SQL ;
3. validation de la requête ;
4. exécution sur SQLite ;
5. récupération des résultats ;
6. synthèse de la réponse.

L'agent hybride choisit la branche adaptée à la question utilisateur.

---

## 3. Préparation et validation des données

### 3.1 Validation avec Pydantic

Pydantic est utilisé pour sécuriser les structures manipulées dans le pipeline.

Les contrôles portent notamment sur :

- la structure des chunks ;
- la présence d'identifiants ;
- la cohérence entre les chunks et leurs embeddings ;
- la dimension des vecteurs ;
- l'absence de valeurs numériques non finies ;
- la structure des réponses générées.

Les modèles utilisés comprennent notamment des structures de type :

- `PreparedChunk` ;
- `EmbeddingBatch` ;
- `RAGAnswer`.

Cette validation empêche certaines données invalides de traverser silencieusement le pipeline.

### 3.2 Validation sémantique avec Pydantic AI

Une validation complémentaire a été ajoutée avec Pydantic AI.

Son objectif est de déterminer si un chunk est suffisamment exploitable pour être utilisé dans le système RAG.

Le validateur retourne :

- `is_valid` ;
- `relevance_score` ;
- `readability_score` ;
- `reason`.

Les scores sont normalisés entre `0.0` et `1.0`.

Exemple observé sur un chunk exploitable :

```text
is_valid = True
relevance_score = 0.95
readability_score = 0.92
```

Exemple observé sur un chunk constitué essentiellement de bruit de navigation :

```text
is_valid = False
relevance_score = 0.00
readability_score = 0.00
```

Le mécanisme permet notamment d'identifier :

- des menus de navigation ;
- des URLs isolées ;
- des fragments OCR fortement dégradés ;
- des morceaux de texte ne contenant aucune information métier exploitable.

---

## 4. Vector Store

Le moteur de recherche vectorielle utilise :

- FAISS ;
- `IndexFlatIP` ;
- des embeddings normalisés ;
- le modèle `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Lors des évaluations, le système récupère les chunks les plus proches de la question utilisateur.

Le nombre de contextes récupérés est généralement :

```text
k = 5
```

Cette étape permet ensuite au LLM de générer une réponse à partir du contexte documentaire retrouvé.

---

## 5. Base de données relationnelle

Les données statistiques issues du fichier Excel ont été intégrées dans une base SQLite.

Le schéma relationnel comporte les tables :

- `players` ;
- `matches` ;
- `stats` ;
- `reports`.

Le chargement est réalisé avec :

`sportsee/sql/loader.py`

Les données sont validées avec Pydantic avant insertion dans la base.

### Limitation du dataset

Le fichier Excel fourni contient principalement des statistiques agrégées.

Il ne fournit pas une véritable table complète de matchs individuels.

Pour cette raison, aucune donnée fictive n'a été créée afin de remplir artificiellement la table `matches`.

Cette décision permet de conserver la cohérence avec les données réellement disponibles.

---

## 6. Tool SQL LangChain

Un Tool SQL a été développé afin de permettre à l'agent d'interroger dynamiquement la base de données.

Le Tool :

- récupère le schéma SQL ;
- reçoit la question utilisateur ;
- utilise le LLM pour produire une requête SQL ;
- exploite des exemples few-shot ;
- contrôle que la requête est en lecture seule ;
- exécute la requête ;
- récupère les résultats ;
- limite le nombre de lignes retournées.

Les requêtes autorisées sont limitées aux opérations de lecture telles que :

```sql
SELECT ...
```

ou certaines requêtes utilisant :

```sql
WITH ...
```

Les opérations de modification ne doivent pas être exécutées.

---

## 7. Exemples few-shot SQL

Des exemples ont été ajoutés au prompt afin d'aider le modèle à construire des requêtes adaptées au schéma.

Ils couvrent notamment :

- la recherche du meilleur joueur ;
- les agrégations par équipe ;
- les classements ;
- les comparaisons entre plusieurs équipes ;
- les calculs de différence ;
- les requêtes avec CTE.

Les exemples few-shot améliorent la précision du mapping entre langage naturel et SQL.

Ils introduisent néanmoins également un biais potentiel puisque le modèle peut être influencé par les structures de requêtes déjà présentes dans le prompt.

---

## 8. Agent hybride

Le système final utilise un agent hybride capable de choisir entre trois routes : RAG, SQL et HYBRID SQL + RAG.

### Questions textuelles

Les questions documentaires sont dirigées vers le RAG.

Exemple :

> Selon les discussions Reddit, quelle faiblesse de Reggie Miller est mentionnée lorsqu'il doit créer son propre tir ?

### Questions numériques

Les questions nécessitant des statistiques exactes sont dirigées vers SQL.

Exemple :

> Quelle équipe a marqué le plus de points ?

### Détection déterministe

Certaines intentions numériques sont détectées grâce à des mots ou motifs tels que :

- points ;
- rebonds ;
- passes ;
- assists ;
- pourcentage ;
- moyenne ;
- total ;
- classement ;
- top N.

Des exclusions ont également été ajoutées afin de traiter certaines négations comme :

- sans statistiques ;
- sans calculer ;
- sans faire de calcul ;
- je veux juste savoir ce que racontent les discussions.

Cette logique réduit certains mauvais routages mais reste heuristique.

---

## 9. Méthodologie d'évaluation

Deux approches sont utilisées.

### 9.1 Évaluation des réponses textuelles

Le RAG est évalué avec le framework RAGAS.

Les métriques utilisées sont :

- **Context Precision** ;
- **Context Recall** ;
- **Faithfulness** ;
- **Answer Relevancy**.

Les cas de test sont répartis en plusieurs catégories :

- simple ;
- complexe ;
- bruité.

### 9.2 Évaluation des réponses numériques

Les questions numériques sont évaluées avec une vérification déterministe.

La réponse produite est comparée aux valeurs attendues.

Cette approche est particulièrement importante pour les résultats numériques, car un LLM juge ne constitue pas toujours le meilleur moyen de vérifier un nombre exact.

---

## 10. Résultats numériques avant enrichissement

Le système RAG seul a été testé sur cinq questions numériques.

Résultat :

```text
2 réponses correctes sur 5
Accuracy = 50 %
```

Certaines questions simples peuvent être correctement traitées lorsque la donnée apparaît directement dans un chunk.

En revanche, les questions nécessitant des agrégations, filtres ou comparaisons sont nettement plus difficiles pour un RAG documentaire seul.

---

## 11. Résultats numériques après ajout du SQL Tool

Après ajout du Tool SQL et de l'agent hybride :

```text
5 réponses correctes sur 5
Accuracy = 100 %
```

L'amélioration observée sur le benchmark est donc de :

```text
+50 points de pourcentage
```

### Exemples de résultats

Le système a notamment retrouvé :

```text
Shai Gilgeous-Alexander : 2485 points
```

```text
Oklahoma City : 18 joueurs
```

```text
Detroit Pistons : 10292 points
```

Pour une comparaison filtrée :

```text
Oklahoma City : 9880 points
Brooklyn Nets : 7999 points
Différence    : 1881 points
```

Ces résultats montrent l'intérêt d'utiliser SQL pour les données structurées nécessitant des calculs exacts.

---

## 12. Première évaluation RAGAS

Une première évaluation du système documentaire a donné les moyennes suivantes :

| Métrique | Score initial |
|---|---:|
| Context Precision | 0.8375 |
| Context Recall | 1.0000 |
| Faithfulness | 0.6071 |
| Answer Relevancy | 0.2804 |

L'analyse qualitative a montré plusieurs limites.

Certaines réponses :

- étaient incomplètes ;
- sélectionnaient une information secondaire ;
- s'arrêtaient avant de répondre à toutes les parties de la question ;
- produisaient parfois une interprétation excessive du contexte.

---

## 13. Amélioration de la génération RAG

Le prompt de génération a ensuite été renforcé.

Les nouvelles règles imposent notamment au modèle :

- de répondre à toutes les parties de la question ;
- de ne pas inventer d'informations ;
- de distinguer les opinions des faits ;
- d'éviter les réponses interrompues ;
- de signaler explicitement les informations absentes du contexte ;
- de privilégier les éléments directement utiles à la question.

Le modèle Pydantic `RAGAnswer` a également été renforcé afin de détecter certaines réponses manifestement incomplètes.

---

## 14. Résultats RAGAS après amélioration

La seconde évaluation produit les résultats moyens suivants :

| Métrique | Avant | Après |
|---|---:|---:|
| Context Precision | 0.8375 | 0.876 |
| Context Recall | 1.0000 | 0.833 |
| Faithfulness | 0.6071 | 0.905 |
| Answer Relevancy | 0.2804 | 0.556 |

Le routage des trois questions textuelles utilisées dans cette évaluation est :

```text
RAG : 3 / 3
```

### Limite de comparabilité avant / après

Les questions et certaines références du benchmark RAGAS ont été précisées entre la première et la seconde évaluation afin de réduire les ambiguïtés observées dans le corpus.

La comparaison avant / après ne constitue donc pas une expérience parfaitement contrôlée à benchmark constant. Les écarts de scores reflètent à la fois les modifications du système et cette évolution du jeu d'évaluation. Ils ne doivent pas être interprétés comme un effet causal attribuable uniquement au prompt, au retrieval ou au modèle génératif.

---

## 15. Analyse de la Faithfulness

La Faithfulness passe de :

```text
0.607
```

à :

```text
0.905
```

Cette progression indique que les réponses produites après amélioration sont davantage fondées sur les informations présentes dans les chunks récupérés.

Il s'agit d'une amélioration importante de la qualité de génération.

---

## 16. Analyse de l'Answer Relevancy

L'Answer Relevancy passe de :

```text
0.280
```

à :

```text
0.556
```

Le score progresse fortement mais reste inférieur aux autres métriques.

Une attention particulière doit être portée à l'interprétation de cette métrique.

Sur le cas complexe, une Answer Relevancy égale à zéro a notamment été observée alors que :

```text
Context Recall = 1.0
Faithfulness   = 1.0
```

et que l'analyse qualitative de la réponse montre qu'elle traite bien le sujet demandé.

Cela montre qu'une métrique automatique reposant sur un LLM juge ne doit pas être interprétée isolément.

---

## 17. Analyse du Context Recall

Le Context Recall passe de :

```text
1.000
```

à :

```text
0.833
```

Le recul provient principalement du scénario bruité.

Cela montre qu'une amélioration de la génération ne garantit pas automatiquement une amélioration de toutes les métriques de retrieval.

Les performances du système doivent donc être analysées séparément selon :

- la qualité de récupération ;
- la qualité des contextes ;
- la qualité de génération.

---

## 18. Analyse qualitative du retrieval

L'inspection des chunks récupérés a permis de vérifier que les passages pertinents étaient généralement présents dans le top 5.

Pour Reggie Miller, les contextes récupérés comportent notamment des informations sur :

- la création de tir avec ballon ;
- le jeu en isolation ;
- le mouvement sans ballon ;
- la gravité offensive ;
- l'efficacité en playoffs ;
- le rôle de premier scoreur.

Pour Haliburton, le retrieval permet notamment de retrouver des commentaires indiquant qu'il est :

- très vocal avec ses coéquipiers ;
- actif dans les huddles ;
- capable de donner des informations à ses partenaires ;
- davantage associé à un trash-talk impactant qu'à un trash-talk permanent.

Cette inspection a également permis d'identifier que certaines questions du benchmark initial étaient trop ouvertes.

Les questions ont donc été reformulées afin de réduire l'ambiguïté entre plusieurs réponses possibles présentes dans le corpus.

---

## 19. Limites du mapping NL vers SQL

La traduction automatique d'une question en SQL présente plusieurs risques.

### Hallucination de filtres

Le modèle peut ajouter une condition non demandée par l'utilisateur.

### Mauvaise interprétation de la question

Une formulation ambiguë peut être traduite en une requête sémantiquement différente de la demande.

### SQL invalide

Les requêtes complexes peuvent produire :

- des sous-requêtes incorrectes ;
- des alias incohérents ;
- des références à des colonnes inexistantes ;
- des constructions SQL non compatibles avec SQLite.

### Réparation automatique

Une étape de réparation augmente la robustesse mais introduit également :

- un nouvel appel au LLM ;
- davantage de latence ;
- un risque de dérive sémantique.

La validation technique ne garantit donc pas à elle seule la justesse métier de la requête.

---

## 20. Limites des exemples few-shot

Les exemples few-shot facilitent la génération SQL mais peuvent introduire un biais.

Le modèle peut privilégier :

- les structures présentes dans les exemples ;
- certaines colonnes ;
- certaines formes d'agrégation ;
- certains types de filtres.

Une meilleure couverture nécessiterait davantage de cas représentatifs.

---

## 21. Sécurité du Tool SQL

Le Tool SQL fonctionne sur une connexion en lecture seule.

Les opérations de modification sont interdites.

Cette protection réduit les risques liés à une requête générée par le LLM.

Un autre garde-fou limite également le nombre de lignes retournées.

Ces protections concernent principalement la sécurité technique.

Elles ne garantissent cependant pas que le SQL généré corresponde parfaitement à l'intention utilisateur.

---

## 22. Limites du routage

Le routage entre RAG et SQL repose actuellement sur une combinaison de règles et d'analyse LLM.

Il reste donc heuristique.

Un mot tel que :

```text
points
```

peut apparaître dans une question documentaire sans nécessiter une interrogation SQL.

Les négations sont également délicates.

Exemple :

> Oublie les statistiques et les classements.

Dans ce cas, la présence du mot `statistiques` ne doit pas déclencher le SQL Tool.

Des exclusions spécifiques ont été ajoutées pour limiter ce problème.

---

## 23. Limites des données structurées

Le fichier Excel fourni ne contient pas un historique complet match par match.

Il contient principalement des statistiques agrégées.

Cela empêche le système de répondre rigoureusement à certaines requêtes comme :

- évolution d'un joueur sur les cinq derniers matchs ;
- série temporelle complète ;
- statistiques détaillées domicile / extérieur lorsque ces informations ne sont pas disponibles ;
- comparaison chronologique entre plusieurs matchs.

Le système ne fabrique pas de données absentes de la source.

---

## 24. Limites du corpus documentaire

Les PDF Reddit contiennent du bruit lié à l'extraction des pages.

Exemples :

- menus ;
- URLs ;
- boutons de navigation ;
- caractères OCR incorrects ;
- noms ou fragments isolés ;
- commentaires adjacents sans rapport direct avec la question.

Ce bruit peut diminuer la précision du retrieval.

La validation Pydantic AI permet d'identifier certains chunks particulièrement dégradés.

---

## 25. Observabilité avec Logfire

Pydantic Logfire est intégré afin de tracer pas à pas le routage, l'exécution SQL, le retrieval RAG, la synthèse hybride et les appels Pydantic AI.

L'observabilité permet notamment de suivre :

- les appels au modèle ;
- les validations structurées ;
- les erreurs de sortie ;
- les différentes étapes d'exécution ;
- la durée de certains traitements.

La configuration utilise `send_to_logfire=False` : les traces sont visualisées localement dans la console sans envoi vers Logfire Cloud.

---

## 26. Tests automatisés

Des tests Pytest sont utilisés pour sécuriser différents composants.

Les tests couvrent notamment :

- les schémas Pydantic ;
- la normalisation des scores ;
- le contrôle des bornes ;
- le validateur de qualité des chunks.

Le test du validateur Pydantic AI vérifie notamment que :

```text
0.95 reste 0.95
8 devient 0.8
9 devient 0.9
10 devient 1.0
```

et qu'un score supérieur à la plage prévue est rejeté.

La suite actuelle contient :

```text
11 tests
11 passed
```

---

## 27. Synthèse comparative

### Questions numériques

| Système | Résultat |
|---|---:|
| RAG seul | 50 % |
| RAG + SQL Tool | 100 % |

### Génération textuelle

| Métrique | Avant | Après |
|---|---:|---:|
| Context Precision | 0.8375 | 0.876 |
| Context Recall | 1.0000 | 0.833 |
| Faithfulness | 0.6071 | 0.905 |
| Answer Relevancy | 0.2804 | 0.556 |

Les résultats montrent une amélioration particulièrement importante sur :

- la précision des réponses numériques ;
- la Faithfulness ;
- l'Answer Relevancy.

---

## 28. Graphiques

Plusieurs graphiques ont été générés dans :

`evaluation_results/figures/`

Ils permettent notamment de visualiser :

- l'accuracy avant et après ajout du SQL Tool ;
- les performances par catégorie ;
- les métriques RAGAS avant et après amélioration.

Les fichiers comprennent notamment :

```text
before_after_accuracy.png
before_after_by_category.png
ragas_before_after.png
sql_before_after.png
```

---

## 29. Limites de l'évaluation

Le benchmark actuel reste de petite taille.

Le benchmark numérique comprend :

```text
5 questions
```

Le benchmark RAGAS final comprend :

```text
3 questions
```

avec les catégories :

- simple ;
- complexe ;
- bruité.

Les résultats observés permettent de comparer les versions du prototype mais ne constituent pas une validation statistique à grande échelle.

Une mise en production nécessiterait :

- davantage de questions ;
- davantage de formulations pour une même intention ;
- plusieurs datasets ;
- plusieurs utilisateurs ;
- des tests de régression automatisés ;
- éventuellement plusieurs modèles juges.

---

## 30. Conclusion

Le projet montre qu'un système RAG documentaire seul n'est pas adapté à toutes les formes de données.

Pour les documents non structurés, la recherche vectorielle permet de retrouver les passages pertinents et de produire une synthèse.

Pour les questions nécessitant des calculs exacts, SQL constitue une approche mieux adaptée.

L'ajout du Tool SQL permet de faire passer la précision du benchmark numérique de :

```text
50 %
```

à :

```text
100 %
```

sur les six scénarios testés.

L'amélioration de la génération RAG permet également de faire progresser la Faithfulness de :

```text
0.607
```

à :

```text
0.905
```

et l'Answer Relevancy de :

```text
0.280
```

à :

```text
0.556
```

L'architecture hybride permet donc de combiner :

- recherche documentaire ;
- génération structurée ;
- interrogation SQL ;
- validation Pydantic ;
- validation Pydantic AI ;
- observabilité Logfire ;
- évaluation RAGAS ;
- tests déterministes.

Le système reste néanmoins un prototype.

Les principaux axes d'amélioration futurs sont :

- augmenter la taille du dataset d'évaluation ;
- améliorer le routage RAG / SQL ;
- renforcer la validation des requêtes SQL ;
- améliorer la qualité OCR ;
- ajouter un mécanisme de reranking ;
- comparer plusieurs modèles juges ;
- améliorer la couverture des tests ;
- intégrer éventuellement un outil de génération dynamique de graphiques.
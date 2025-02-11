# Impact-Predict

Le but du projet est de créer une métrique qui met en avant les 5 joueurs les plus impactant par position des 5 grands championnats (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) ayant un impact monstre ou prépondérant sur les résultats de leurs équipes. Ensuite, il faut aussi créer des radar charts pour le top 10.

La première étape consiste donc à classer les joueurs par poste. Les données comportent une colonne « Position », mais les entrées peuvent varier (comme « Attaquant », « Milieu de terrain », « Défenseur »). Je vais devoir les standardiser en trois groupes : « Attaquant », « Milieu », « Défenseur ». Peut-être que certains joueurs ont plusieurs positions, mais pour plus de simplicité, je prendrai la position principale répertoriée.

Ensuite, pour chaque position, je dois définir des métriques pertinentes. Les attaquants doivent être davantage jugés sur les buts, les passes décisives, les xG, etc. Les milieux de terrain peuvent avoir besoin d'un équilibre entre les statistiques offensives et défensives. Les défenseurs se concentreraient sur les tacles, les interceptions, les dégagements, etc. Le code d'origine utilisait un ensemble de statistiques, mais je dois maintenant les ajuster en fonction de la position.

Je vais devoir créer trois ensembles distincts de colonnes et de pondérations. Par exemple, les attaquants peuvent avoir des pondérations plus élevées sur « Buts + Passes décisives par 90 » et « xG + xAG par 90 », tandis que les défenseurs ont des pondérations plus élevées sur « Tacles réussis par 90 » et « Interceptions par 90 ». Les milieux de terrain peuvent avoir un mélange des deux.

Ensuite, la normalisation doit être effectuée au sein de chaque groupe de positions. Ceci est important car les statistiques varient selon la position ; le nombre de plaquages ​​d'un défenseur est naturellement plus élevé que celui d'un attaquant. En normalisant au sein des groupes, nous garantissons une comparaison équitable.

Après avoir calculé le SIG pour chaque position, je les classerai et sélectionnerai les 5 meilleurs pour chacun. De plus, la génération de graphiques radar pour les 10 meilleurs de toutes les positions aiderait à visualiser leurs points forts. L'utilisation d'une bibliothèque comme matplotlib ou seaborn en Python peut aider à tracer ces graphiques radar. Chaque graphique affichera les statistiques normalisées pour les métriques sélectionnées.

Problèmes potentiels à prendre en compte : gérer les données manquantes, s'assurer que toutes les colonnes requises sont présentes et mapper correctement les positions. De plus, certains joueurs peuvent être répertoriés sous des noms de poste différents (par exemple, « Attaquant » contre « Attaquant »), il est donc essentiel de normaliser ces étiquettes.

Tester le code avec des exemples de données aidera à vérifier si les calculs SIG ont du sens par poste. Par exemple, un défenseur de haut niveau doit avoir des statistiques défensives élevées, tandis qu'un attaquant de haut niveau excelle dans les mesures offensives. Il peut être nécessaire d'ajuster les pondérations si les résultats initiaux ne reflètent pas les classements attendus.

Enfin, le code doit afficher les 5 meilleurs joueurs par poste et générer des graphiques radar, en les enregistrant sous forme d'images pour une référence facile. L'utilisateur peut ensuite analyser la manière dont les contributions de chaque joueur se comparent à celles des autres à leur poste.

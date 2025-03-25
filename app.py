import streamlit as st
import pandas as pd
import numpy as np
from soccerplots.radar_chart import Radar
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration de la page
st.set_page_config(page_title="Player Impact Analysis", layout="wide")

# Fonction de chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("df_Big2025.csv")
    df = df[df['Pourcentage de minutes jouees'] > 60]

    # Nettoyage des données
    df = df[df['Ligue'].isin(['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'])]
    df['Position'] = df['Position'].replace({
        'Forward': 'Attaquant',
        'Midfielder': 'Milieu',
        'Defender': 'Défenseur'
    })
    df = df[df['Position'].notna()]
    
    # Calcul des stats par 90 minutes
    def per90(col, minutes):
        return (col / minutes) * 90
    
    stats = [
    'Buts', 'Passes decisives', 'Passes decisives attendues (xAG)',
    'Tacles reussis', 'Interceptions', 'Erreurs menant a un tir',
    'Duels aeriens gagnes', 'Courses progressives', 'Ballons recuperes',
    'Passes progressives', 'Dribbles reussis', 'Passes cles', 'Passes dans le dernier tiers',
    'Pourcentage de duels gagnes', 'Touches de balle surface offensive', 'Degagements',
    'Fautes commises', 'Passes longues reussies', 'Pourcentage de passes reussies'
]
    for stat in stats:
        df[f'{stat} par 90'] = per90(df[stat], df['Minutes jouees'])
    
    return df

# Chargement des données
df = load_data()

# Configuration des caractéristiques par position (10 variables)
position_config = {
    'Attaquant': {
        'features': [
            'Buts par 90',
            'Passes decisives par 90',
            'Buts attendus par 90 minutes',
            'Dribbles reussis par 90',
            'Courses progressives par 90',
            'Tirs cadres par 90 minutes',
            'Passes cles par 90',
            'Pourcentage de duels gagnes par 90',
            'Touches de balle surface offensive par 90',
            'Passes decisives attendues par 90 minutes'
        ],
        'weights': [0.18, 0.15, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02]
    },
    'Milieu': {
        'features': [
            'Passes progressives par 90',
            'Passes decisives par 90',
            'Ballons recuperes par 90',
            'Interceptions par 90',
            'Dribbles reussis par 90',
            'Buts et passes attendus par 90 minutes',
            'Passes dans le dernier tiers par 90',
            'Duels aeriens gagnes par 90',
            'Courses progressives par 90',
            'Pourcentage de passes reussies par 90'
        ],
        'weights': [0.15, 0.15, 0.13, 0.12, 0.10, 0.10, 0.08, 0.07, 0.06, 0.04]
    },
    'Défenseur': {
        'features': [
            'Tacles reussis par 90',
            'Interceptions par 90',
            'Duels aeriens gagnes par 90',
            'Ballons recuperes par 90',
            'Passes progressives par 90',
            'Pourcentage de passes reussies par 90',
            'Degagements par 90',
            'Fautes commises par 90',
            'Passes longues reussies par 90',
            'Erreurs menant a un tir par 90'
        ],
        'weights': [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]
    }
}

# Calcul des scores d'impact
@st.cache_data
def calculate_impact_scores(df):
    scores_df = pd.DataFrame()
    
    for position in position_config:
        config = position_config[position]
        pos_df = df[df['Position'] == position].copy()
        
        # Normalisation
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(pos_df[config['features']])
        
        # Calcul du score
        pos_df['Impact Score'] = np.dot(scaled_features, config['weights'])
        
        # Conservation des features normalisées pour les visualisations
        for i, feature in enumerate(config['features']):
            pos_df[f'Scaled {feature}'] = scaled_features[:,i]
        
        scores_df = pd.concat([scores_df, pos_df])
    
    return scores_df

# Application Streamlit
def main():
    st.title("🔝 Analyse d'Impact des Joueurs - Top 5 Championnats 🔝")
    
    # Calcul des scores
    df_scored = calculate_impact_scores(df)

    # Sidebar - Sélections
    with st.sidebar:
        # Sélection du championnat
        league = st.selectbox(
            'Sélectionnez un championnat',
            ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
        )
        
        filtered_df = df_scored[df_scored['Ligue'] == league]

        # Options d'affichage
        st.markdown("---")
        show_top5 = st.checkbox("Afficher les Top 5 par position", value=True)
        enable_comparison = st.checkbox("Activer la comparaison de joueurs", value=False)
        
        # Comparaison de joueurs (nouvelle version)
        if enable_comparison:
            st.markdown("## Comparaison de Joueurs")
            
            # Sélection de la position commune
            selected_position = st.selectbox(
                'Sélectionnez la position',
                ['Attaquant', 'Milieu', 'Défenseur']
            )
            
            # Sélection des joueurs avec championnat individuel
            selected_players = []
            selected_leagues = []
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Joueur 1**")
                league1 = st.selectbox(
                    'Championnat 1',
                    ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'],
                    key='league1'
                )
                players1 = df_scored[
                    (df_scored['Position'] == selected_position) & 
                    (df_scored['Ligue'] == league1)
                ]['Joueur'].unique()
                player1 = st.selectbox('Sélection', players1, key='player1')
                selected_players.append(player1)
                selected_leagues.append(league1)

            with cols[1]:
                st.markdown("**Joueur 2**")
                league2 = st.selectbox(
                    'Championnat 2',
                    ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'],
                    key='league2'
                )
                players2 = df_scored[
                    (df_scored['Position'] == selected_position) & 
                    (df_scored['Ligue'] == league2)
                ]['Joueur'].unique()
                player2 = st.selectbox('Sélection', players2, key='player2')
                selected_players.append(player2)
                selected_leagues.append(league2)

    
    
    # Affichage des top 5 par position (seulement si activé)
    if show_top5:
        positions = ['Attaquant', 'Milieu', 'Défenseur']
        cols = st.columns(3)
        
        for i, position in enumerate(positions):
            with cols[i]:
                st.subheader(f"Top 5 {position}")
                pos_df = filtered_df[filtered_df['Position'] == position] \
                    .sort_values('Impact Score', ascending=False) \
                    .head(5)
                
                for _, row in pos_df.iterrows():
                    with st.expander(f"{row['Joueur']} ({row['Equipe']}, {int(row['Age'])})"):
                        st.metric("Score d'Impact", f"{row['Impact Score']:.2f}")
                        
                        # Radar chart
                        features = position_config[position]['features']
                        scaled_features = [f'Scaled {f}' for f in features]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=row[scaled_features].values,
                            theta=features,
                            fill='toself',
                            line_color='blue'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[-10, 10]
                                )),
                            showlegend=False,
                            height=300,
                            width=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

    # Section de comparaison modifiée avec radar unique
    if enable_comparison and len(selected_players) == 2:
        st.markdown("---")
        st.subheader("🔍 Comparaison Joueurs")
    
        # Récupération des données des joueurs
        player1 = df_scored[
            (df_scored['Joueur'] == selected_players[0]) & 
            (df_scored['Ligue'] == selected_leagues[0])
        ].iloc[0]
        
        player2 = df_scored[
            (df_scored['Joueur'] == selected_players[1]) & 
            (df_scored['Ligue'] == selected_leagues[1])
        ].iloc[0]

        # Vérification du même poste
        if player1['Position'] != player2['Position']:
            st.error("Les joueurs doivent avoir le même poste pour être comparés !")
            st.stop()

        # Création des colonnes pour les métriques
        col1, col2 = st.columns(2)
    
        # Modifiez les sections des joueurs comme ceci :
        with col1:
            st.markdown(f"### {player1['Joueur']}")
            st.write(f"**Équipe:** {player1['Equipe']} ({player1['Ligue']})")
            st.write(f"**Âge:** {int(player1['Age'])}")
            st.metric("Score d'Impact", f"{player1['Impact Score']:.2f}")
            
        with col2:
            st.markdown(f"### {player2['Joueur']}")
            st.write(f"**Équipe:** {player2['Equipe']} ({player2['Ligue']})")
            st.write(f"**Âge:** {int(player2['Age'])}")
            st.metric("Score d'Impact", f"{player2['Impact Score']:.2f}")

        # Création du radar chart combiné
        #position = player1['Position']
        #features = position_config[position]['features']
        #scaled_features = [f'Scaled {f}' for f in features]

        #fig = go.Figure()
    
        # Trace pour le joueur 1
        #fig.add_trace(go.Scatterpolar(
        #    r=player1[scaled_features].values,
        #    theta=features,
        #    name=player1['Joueur'],
        #    fill='toself',
        #    line_color='#1f77b4'  # Bleu
        #))
    
        # Trace pour le joueur 2
        #fig.add_trace(go.Scatterpolar(
        #    r=player2[scaled_features].values,
        #    theta=features,
        #    name=player2['Joueur'],
        #    fill='toself',
        #    line_color='#ff7f0e'  # Orange
        #))

        # Mise en forme du radar
        #fig.update_layout(
        #    polar=dict(
        #        radialaxis=dict(
        #            visible=True,
        #            range=[-10, 10],
        #            tickfont=dict(size=8)
        #    )),
        #    legend=dict(
        #        orientation="h",
        #        yanchor="bottom",
        #        y=1.1,
        #        xanchor="center",
        #        x=0.5
        #    ),
        #    margin=dict(l=50, r=50, t=50, b=50),
        #    height=500,
        #)

        # Note de bas de page
        endnote = "Source : FBref | Made by : Moubarak Issa"


        # Création du radar chart
        position = player1['Position']
        features = position_config[position]['features']
    
        # Normalisation des valeurs sur 100
        max_values = df[features].max()  # On utilise le max du dataset complet
        player1_data = [(player1[feat] / max_values[feat]) * 100 for feat in features]
        player2_data = [(player2[feat] / max_values[feat]) * 100 for feat in features]

        # Configuration du radar
        radar = Radar(
            background_color="#121212",
            patch_color="#28252C", 
            label_color="#F0FFF0",
            range_color="#F0FFF0",
            label_fontsize=10    
        )

        # Création de la figure avec une taille réduite
        #fig, ax = plt.subplots(figsize=(7, 7))  # Ajuster la taille globale ici


        fig, ax = radar.plot_radar(
            ranges=[(0, 100)] * len(features),
            params=features,
            values=[player1_data, player2_data],
            radar_color=['#9B3647', '#3282b8'],
            #title=f"Profil de performance - {position}",
            endnote=endnote,
            alphas=[0.55, 0.5],
            compare=True,
        )
        fig.set_size_inches(8, 8)

        # Affichage du radar
        st.markdown("### 📊 Profil comparé (par 90 minutes)")
        #st.plotly_chart(fig, use_container_width=True)
        # Ajustement de la mise en page
        plt.tight_layout(pad=3.0)
        # Affichage du radar dans Streamlit
        st.pyplot(fig)

        # Tableau comparatif
        st.markdown("### 📊 Comparaison détaillée")
        comparison_df = pd.DataFrame({
            'Statistique': position_config[player1['Position']]['features'],
            player1['Joueur']: player1[position_config[player1['Position']]['features']],
            player2['Joueur']: player2[position_config[player2['Position']]['features']]
        })
        
        # Formatage des nombres
        for col in comparison_df.columns[1:]:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            comparison_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Statistique": "Statistique",
                player1['Joueur']: player1['Joueur'],
                player2['Joueur']: player2['Joueur']
            }
        )


if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration de la page
st.set_page_config(page_title="Player Impact Analysis", layout="wide")

# Fonction de chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("df_Big5.csv")
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
    'Buts', 'Passes decisives', 'Tacles reussis', 'Interceptions', 
    'Duels aeriens gagnes', 'Courses progressives', 'Ballons recuperes',
    'Passes progressives', 'Passes reussies totales', # Ajouté
]
    for stat in stats:
        df[f'{stat} par 90'] = per90(df[stat], df['Minutes jouees'])
    
    return df

# Chargement des données
df = load_data()

# Configuration des caractéristiques par position
position_config = {
    'Attaquant': {
        'features': [
            'Buts par 90',
            'Passes decisives par 90',
            'xG par 90 minutes',
            'Dribbles reussis',
            'Courses progressives par 90'
        ],
        'weights': [0.3, 0.25, 0.25, 0.1, 0.1]
    },
    'Milieu': {
        'features': [
            'Passes decisives par 90',
            'Passes progressives par 90',
            'Tacles reussis par 90',
            'Interceptions par 90',
            'xG + xAG par 90 minutes'
        ],
        'weights': [0.25, 0.25, 0.2, 0.2, 0.1]
    },
    'Défenseur': {
        'features': [
            'Tacles reussis par 90',
            'Interceptions par 90',
            'Duels aeriens gagnes par 90',
            'Ballons recuperes par 90',
            'Passes reussies totales par 90'
        ],
        'weights': [0.3, 0.3, 0.2, 0.1, 0.1]
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
    
    # Sidebar - Sélections
    with st.sidebar:
        # Sélection du championnat
        league = st.selectbox(
            'Sélectionnez un championnat',
            ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
        )
        
        # Options d'affichage
        st.markdown("---")
        show_top5 = st.checkbox("Afficher les Top 5 par position", value=True)
        enable_comparison = st.checkbox("Activer la comparaison de joueurs", value=False)
        
        # Comparaison de joueurs (seulement si activé)
        selected_players = []
        if enable_comparison:
            st.markdown("## Comparaison de Joueurs")
            df_scored = calculate_impact_scores(df)
            filtered_df = df_scored[df_scored['Ligue'] == league]
            players_list = filtered_df['Joueur'].unique().tolist()
            selected_players = st.multiselect(
                'Choisissez 2 joueurs à comparer',
                players_list,
                max_selections=2
            )

    # Calcul des scores
    df_scored = calculate_impact_scores(df)
    filtered_df = df_scored[df_scored['Ligue'] == league]
    
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
                                    range=[-3, 3]
                                )),
                            showlegend=False,
                            height=300,
                            width=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

    # Section de comparaison (seulement si activé et 2 joueurs sélectionnés)
    if enable_comparison and len(selected_players) == 2:
        st.markdown("---")
        st.subheader("🔍 Comparaison côte à côte")
        # Récupération des données des joueurs
        player1 = filtered_df[filtered_df['Joueur'] == selected_players[0]].iloc[0]
        player2 = filtered_df[filtered_df['Joueur'] == selected_players[1]].iloc[0]
        
        # Création des colonnes pour l'affichage
        col1, col2 = st.columns(2)
        
        # Joueur 1
        with col1:
            st.markdown(f"### {player1['Joueur']}")
            st.write(f"**Équipe:** {player1['Equipe']}")
            st.write(f"**Âge:** {int(player1['Age'])}")
            st.write(f"**Position:** {player1['Position']}")
            st.metric("Score d'Impact", f"{player1['Impact Score']:.2f}")
            
            # Radar chart
            features = position_config[player1['Position']]['features']
            scaled_features = [f'Scaled {f}' for f in features]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=player1[scaled_features].values,
                theta=features,
                fill='toself',
                line_color='blue',
                name=player1['Joueur']
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Joueur 2
        with col2:
            st.markdown(f"### {player2['Joueur']}")
            st.write(f"**Équipe:** {player2['Equipe']}")
            st.write(f"**Âge:** {int(player2['Age'])}")
            st.write(f"**Position:** {player2['Position']}")
            st.metric("Score d'Impact", f"{player2['Impact Score']:.2f}")
            
            # Radar chart
            features = position_config[player2['Position']]['features']
            scaled_features = [f'Scaled {f}' for f in features]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=player2[scaled_features].values,
                theta=features,
                fill='toself',
                line_color='red',
                name=player2['Joueur']
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

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
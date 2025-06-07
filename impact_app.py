import streamlit as st
import pandas as pd
import numpy as np
from soccerplots.radar_chart import Radar
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration de la page avec thème personnalisé
st.set_page_config(
    page_title="Player Impact Analysis", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚽"
)

# CSS personnalisé pour améliorer le style
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --border-color: #e5e7eb;
    }
    
    /* Style général */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.15);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Cards pour les joueurs */
    .player-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .player-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-color: var(--secondary-color);
    }
    
    .player-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(45deg, var(--accent-color), var(--secondary-color));
    }
    
    .player-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .player-info {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    /* Métriques personnalisées */
    .custom-metric {
        background: linear-gradient(135deg, var(--accent-color), #059669);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Position badges */
    .position-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .position-attaquant {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    
    .position-milieu {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #60a5fa;
    }
    
    .position-defenseur {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #4ade80;
    }
    
    /* Section headers */
    .section-header {
        border-left: 4px solid var(--secondary-color);
        padding: 1rem 0 1rem 1.5rem;
        margin: 2rem 0 1rem 0;
        background: var(--bg-secondary);
        border-radius: 0 8px 8px 0;
    }
    
    .section-header h2 {
        margin: 0;
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-secondary);
    }
    
    /* Comparaison styling */
    .comparison-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid var(--border-color);
    }
    
    .vs-divider {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary-color);
        margin: 1rem 0;
    }
    
    /* Tableau amélioré */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

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

# Configuration des caractéristiques par position
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
        'weights': [0.18, 0.15, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02],
        'color': '#f59e0b',
        'icon': '⚽'
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
        'weights': [0.15, 0.15, 0.13, 0.12, 0.10, 0.10, 0.08, 0.07, 0.06, 0.04],
        'color': '#3b82f6',
        'icon': '🎯'
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
        'weights': [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02],
        'color': '#10b981',
        'icon': '🛡️'
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

def create_player_card(player_data, position_config):
    """Créer une carte joueur stylisée"""
    position = player_data['Position']
    config = position_config[position]
    
    position_class = f"position-{position.lower()}"
    
    card_html = f"""
    <div class="player-card fade-in">
        <div class="player-name">
            {config['icon']} {player_data['Joueur']}
        </div>
        <div class="player-info">
            <span class="position-badge {position_class}">{position}</span>
            <br>
            <strong>{player_data['Equipe']}</strong> • {int(player_data['Age'])} ans
        </div>
        <div class="custom-metric">
            <div class="metric-value">{player_data['Impact Score']:.2f}</div>
            <div class="metric-label">Score d'Impact</div>
        </div>
    </div>
    """
    return card_html

def create_radar_chart(player_data, features, scaled_features, title=""):
    """Créer un radar chart amélioré"""
    fig = go.Figure()
    
    # Couleurs selon la position
    position = player_data['Position']
    color = position_config[position]['color']
    
    fig.add_trace(go.Scatterpolar(
        r=player_data[scaled_features].values,
        theta=[f.replace(' par 90', '').replace(' par 90 minutes', '') for f in features],
        fill='toself',
        line=dict(color=color, width=2),
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
        name=player_data['Joueur']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-3, 3],
                tickfont=dict(size=10, color='#6b7280'),
                gridcolor='rgba(107, 114, 128, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color='#374151')
            )
        ),
        showlegend=False,
        height=400,
        title=dict(
            text=title,
            font=dict(size=16, color='#1f2937'),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Application Streamlit principale
def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>⚽ Analyse d'Impact des Joueurs</h1>
        <p>Découvrez les performances des meilleurs joueurs des Top 5 Championnats européens</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    df_scored = calculate_impact_scores(df)

    # Sidebar améliorée
    with st.sidebar:
        st.markdown("### 🎛️ Paramètres d'analyse")
        
        # Sélection du championnat
        league = st.selectbox(
            '🏆 Championnat',
            ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'],
            help="Sélectionnez le championnat à analyser"
        )
        
        filtered_df = df_scored[df_scored['Ligue'] == league]
        
        st.markdown("---")
        
        # Options d'affichage
        st.markdown("### 📊 Options d'affichage")
        show_top5 = st.checkbox("Top 5 par position", value=True)
        show_stats = st.checkbox("Statistiques détaillées", value=False)
        enable_comparison = st.checkbox("Comparaison de joueurs", value=False)
        
        if enable_comparison:
            st.markdown("---")
            st.markdown("### 🔍 Comparaison")
            
            # Sélection de la position
            selected_position = st.selectbox(
                '📍 Position',
                ['Attaquant', 'Milieu', 'Défenseur'],
                help="Les joueurs doivent avoir la même position pour être comparés"
            )
            
            # Sélection des joueurs
            selected_players = []
            selected_leagues = []
            
            for i in range(2):
                st.markdown(f"**Joueur {i+1}**")
                league_key = f'league{i+1}'
                player_key = f'player{i+1}'
                
                league_i = st.selectbox(
                    f'Championnat {i+1}',
                    ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'],
                    key=league_key
                )
                
                players_i = df_scored[
                    (df_scored['Position'] == selected_position) & 
                    (df_scored['Ligue'] == league_i)
                ]['Joueur'].unique()
                
                player_i = st.selectbox(
                    f'Sélection {i+1}', 
                    players_i, 
                    key=player_key
                )
                
                selected_players.append(player_i)
                selected_leagues.append(league_i)

    # Contenu principal
    if show_top5:
        st.markdown('<div class="section-header"><h2>🏆 Top 5 par Position</h2></div>', 
                   unsafe_allow_html=True)
        
        positions = ['Attaquant', 'Milieu', 'Défenseur']
        tabs = st.tabs([f"{position_config[pos]['icon']} {pos}" for pos in positions])
        
        for i, position in enumerate(positions):
            with tabs[i]:
                pos_df = filtered_df[filtered_df['Position'] == position] \
                    .sort_values('Impact Score', ascending=False) \
                    .head(5)
                
                # Affichage des joueurs en colonnes
                cols = st.columns(2)
                for idx, (_, row) in enumerate(pos_df.iterrows()):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        # Carte joueur
                        st.markdown(create_player_card(row, position_config), 
                                  unsafe_allow_html=True)
                        
                        # Radar chart
                        features = position_config[position]['features']
                        scaled_features = [f'Scaled {f}' for f in features]
                        
                        fig = create_radar_chart(
                            row, features, scaled_features,
                            f"Profil de {row['Joueur']}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if show_stats:
                            with st.expander("📈 Statistiques détaillées"):
                                stats_df = pd.DataFrame({
                                    'Statistique': features,
                                    'Valeur': [f"{row[f]:.2f}" for f in features]
                                })
                                st.dataframe(stats_df, hide_index=True)
                        
                        st.markdown("---")

    # Section comparaison améliorée
    if enable_comparison and len(selected_players) == 2:
        st.markdown('<div class="section-header"><h2>⚔️ Comparaison de Joueurs</h2></div>', 
                   unsafe_allow_html=True)
        
        # Récupération des données
        player1 = df_scored[
            (df_scored['Joueur'] == selected_players[0]) & 
            (df_scored['Ligue'] == selected_leagues[0])
        ].iloc[0]
        
        player2 = df_scored[
            (df_scored['Joueur'] == selected_players[1]) & 
            (df_scored['Ligue'] == selected_leagues[1])
        ].iloc[0]
        
        if player1['Position'] != player2['Position']:
            st.error("⚠️ Les joueurs doivent avoir le même poste pour être comparés !")
            st.stop()
        
        # Interface de comparaison
        st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([5, 1, 5])
        
        with col1:
            st.markdown(create_player_card(player1, position_config), 
                       unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown(create_player_card(player2, position_config), 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Radar chart comparatif
        position = player1['Position']
        features = position_config[position]['features']
        
        # Création du radar chart avec mplsoccer
        scaler = StandardScaler()
        df_normalized = scaler.fit_transform(df_scored[features])
        
        player1_idx = df_scored[
            (df_scored['Joueur'] == selected_players[0]) & 
            (df_scored['Ligue'] == selected_leagues[0])
        ].index[0]
        
        player2_idx = df_scored[
            (df_scored['Joueur'] == selected_players[1]) & 
            (df_scored['Ligue'] == selected_leagues[1])
        ].index[0]
        
        player1_data = df_normalized[df_scored.index.get_loc(player1_idx)]
        player2_data = df_normalized[df_scored.index.get_loc(player2_idx)]
        
        try:
            radar = Radar(
                label_fontsize=10,
                range_color="#F0FFF0",
                label_color="white",
                patch_color="#28252C",
                background_color="#121212"
            )
            
            endnote = "Source : FBref | Made by : Moubarak Issa"
            
            fig, ax = radar.plot_radar(
                ranges=[(0, 1)] * len(features),
                params=[f.replace(' par 90', '').replace(' par 90 minutes', '') for f in features],
                values=[player1_data, player2_data],
                radar_color=['#9B3647', '#3282b8'],
                endnote=endnote,
                alphas=[0.6, 0.4],
                compare=True
            )
            
            # Légende améliorée
            labels = [player1['Joueur'], player2['Joueur']]
            colors = ['#9B3647', '#3282b8']
            patches = [plt.Rectangle((0,0),1,1, color=colors[i], ec="black") for i in range(2)]
            ax.legend(
                handles=patches,
                labels=labels,
                loc='upper right',
                bbox_to_anchor=(1.25, 1),
                fontsize=10,
                frameon=True,
                facecolor='white',
                edgecolor='gray'
            )
            
            fig.set_size_inches(10, 10)
            plt.tight_layout(pad=3.0)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erreur lors de la création du radar chart : {e}")
            
            # Fallback avec Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=player1_data,
                theta=[f.replace(' par 90', '') for f in features],
                name=player1['Joueur'],
                fill='toself',
                line_color='#9B3647'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=player2_data,
                theta=[f.replace(' par 90', '') for f in features],
                name=player2['Joueur'],
                fill='toself',
                line_color='#3282b8'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau comparatif stylisé
        st.markdown("### 📊 Comparaison Statistique Détaillée")
        
        comparison_df = pd.DataFrame({
            'Statistique': [f.replace(' par 90', '').replace(' par 90 minutes', '') 
                          for f in features],
            player1['Joueur']: [f"{player1[f]:.2f}" for f in features],
            player2['Joueur']: [f"{player2[f]:.2f}" for f in features],
            'Différence': [f"{player1[f] - player2[f]:+.2f}" for f in features]
        })
        
        st.dataframe(
            comparison_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Statistique": st.column_config.TextColumn("📊 Statistique", width="medium"),
                player1['Joueur']: st.column_config.NumberColumn(
                    f"🔵 {player1['Joueur']}", 
                    format="%.2f"
                ),
                player2['Joueur']: st.column_config.NumberColumn(
                    f"🔴 {player2['Joueur']}", 
                    format="%.2f"
                ),
                "Différence": st.column_config.NumberColumn(
                    "📈 Écart", 
                    format="%+.2f",
                    help="Différence entre les deux joueurs (positif = avantage joueur 1)"
                )
            }
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem; margin-top: 3rem;">
        <p>📊 <strong>Source des données :</strong> FBref | 
        👨‍💻 <strong>Développé par :</strong> Moubarak Issa | 
        ⚡ <strong>Propulsé par :</strong> Streamlit</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">
            Analyse basée sur les performances des joueurs ayant joué plus de 60% du temps de jeu
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
import customtkinter as ctk
import pandas as pd
from lck_data_pipeline import LCKDataPipeline
from lck_ml_model import LCKPredictor, BettingCalculator
from lck_config import BANKROLL

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class LCKPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LCK Prediction OS v2.1 — 2026 Season")
        self.geometry("950x700")

        # Motores internos
        self.pipeline = LCKDataPipeline()
        self.predictor = LCKPredictor()
        self.calculator = BettingCalculator()
        self.df_stats = pd.DataFrame()

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="LCK ANALYTICS", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=20)

        self.btn_update = ctk.CTkButton(self.sidebar, text="Actualizar Datos API", command=self.update_data)
        self.btn_update.grid(row=1, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Estado: Offline", font=("Consolas", 12))
        self.status_label.grid(row=2, column=0, padx=20, pady=10)

        # --- Panel Principal ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        ctk.CTkLabel(self.main_frame, text="Análisis de Partida y Value Bet", font=("Segoe UI", 22, "bold")).pack(pady=20)

        # Contenedor de Selección y Cuota
        self.input_container = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_container.pack(pady=10)

        # Equipo A
        self.team_a_combo = ctk.CTkComboBox(self.input_container, values=["Cargar datos..."], width=180)
        self.team_a_combo.grid(row=0, column=0, padx=10)

        ctk.CTkLabel(self.input_container, text="VS", font=("Segoe UI", 14, "bold")).grid(row=0, column=1)

        # Equipo B
        self.team_b_combo = ctk.CTkComboBox(self.input_container, values=["Cargar datos..."], width=180)
        self.team_b_combo.grid(row=0, column=2, padx=10)

        # Campo para Cuota Codere
        ctk.CTkLabel(self.main_frame, text="Cuota Codere (ej: 1.95):", font=("Segoe UI", 12)).pack(pady=(15, 0))
        self.odd_entry = ctk.CTkEntry(self.main_frame, placeholder_text="1.00", width=100, justify="center")
        self.odd_entry.pack(pady=5)
        self.odd_entry.insert(0, "2.00") # Valor inicial

        self.btn_predict = ctk.CTkButton(self.main_frame, text="CALCULAR PREDICCIÓN", 
                                         command=self.predict_match, 
                                         fg_color="#1f538d", height=40, font=("Segoe UI", 13, "bold"))
        self.btn_predict.pack(pady=20)

        # Resultados Visuales
        self.res_label = ctk.CTkLabel(self.main_frame, text="---", font=("Consolas", 16), 
                                      fg_color="#1a1a1a", corner_radius=10, pady=15)
        self.res_label.pack(padx=40, fill="x")

        # Terminal de Value Bet
        self.textbox_bets = ctk.CTkTextbox(self.main_frame, font=("Consolas", 13), border_width=2)
        self.textbox_bets.pack(padx=40, pady=20, fill="both", expand=True)

    def update_data(self):
        self.status_label.configure(text="⏳ Descargando...")
        self.update_idletasks()

        df_matches = self.pipeline.get_all_matches(limit=100) #
        if df_matches.empty:
            self.status_label.configure(text="❌ Error API")
            return

        self.df_stats = self.pipeline.build_team_stats(df_matches) #
        self.predictor.train(df_matches) #
        
        team_names = sorted(self.df_stats["team_name"].unique().tolist())
        self.team_a_combo.configure(values=team_names)
        self.team_b_combo.configure(values=team_names)
        
        self.status_label.configure(text="✅ Datos Listos")

    def predict_match(self):
        if self.df_stats.empty: return

        try:
            name_a = self.team_a_combo.get()
            name_b = self.team_b_combo.get()
            odd_input = float(self.odd_entry.get())

            # Extraer KPIs para el modelo
            stats_a = self.df_stats[self.df_stats["team_name"] == name_a].iloc[0].to_dict()
            stats_b = self.df_stats[self.df_stats["team_name"] == name_b].iloc[0].to_dict()

            prediction = self.predictor.predict_match(stats_a, stats_b) #

            self.res_label.configure(text=(
                f"Ganador Probable: {prediction['predicted_winner']}\n"
                f"Confianza: {prediction['confidence']*100:.1f}%"
            ))

            # Análisis de apuesta real con el bankroll configurado
            analysis = self.calculator.analyze_bet(name_a, name_b, prediction['prob_a'], odd_input, BANKROLL)
            
            self.display_bet(analysis)

        except ValueError:
            self.textbox_bets.delete("0.0", "end")
            self.textbox_bets.insert("0.0", "ERROR: Ingresa una cuota válida (numérica).")

    def display_bet(self, a):
        text = f" ANÁLISIS DE VALOR (Kelly Criterion)\n"
        text += f" {'='*40}\n"
        text += f" Match: {a['team_a']} vs {a['team_b']}\n"
        text += f" Prob. Modelo: {a['prob_model']}%\n"
        text += f" Prob. Codere: {a['implied_prob']}%\n"
        text += f" Ventaja (Edge): {a['edge_value']}%\n"
        text += f" {'-'*40}\n"
        text += f" RESULTADO: {a['tag']}\n"
        
        if a['is_value_bet']:
            text += f" SUGERENCIA: Invertir ${a['stake_mxn']} MXN\n"
            text += f" RETORNO ESPERADO: +${a['expected_return']} MXN\n"
        else:
            text += f" SUGERENCIA: No entrar a esta posición.\n"
        
        self.textbox_bets.delete("0.0", "end")
        self.textbox_bets.insert("0.0", text)

if __name__ == "__main__":
    app = LCKPredictorApp()
    app.mainloop()
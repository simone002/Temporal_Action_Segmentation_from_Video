# Temporal_Action_Segmentation_from_Video

Traccia 13: Segmentazione Temporale delle Azioni da Video

Dimensione consigliata: Piccola
Modulo di riferimento: Comprensione Video

Descrizione del problema

Mentre il riconoscimento delle azioni identifica cosa accade in un singolo clip video, la segmentazione temporale funziona come un editor preciso che individua esattamente quando ogni azione inizia e finisce all’interno di un flusso video lungo e non strutturato.

In questo task, devi mappare sequenze di feature video (vettori) su lunghe sequenze temporali per predire, in modo denso (frame per frame o segmento per segmento), lo stato dell’attività, determinando con precisione i confini delle azioni umane.

Dataset

Userai il dataset EGTEA Gaze+
(https://cbs.ic.gatech.edu/fpv/
)

Usa feature già estratte (es. da protocolli RULSTM)
Sono inclusi timestamp annotati di inizio e fine azione

- Obiettivi Minimi

🔹 Baseline CNN Temporale
Usa una CNN 1D lungo la dimensione temporale
Input: sequenze di feature
Output: predizione dello stato dell’azione per ogni frame/segmento

🔹 Modulo RNN
Implementa una LSTM
Deve catturare:
contesto temporale locale
dipendenze nel tempo (memoria)

🔹 Architettura xLSTM
Versione più recente e avanzata della LSTM
Progettata per gestire sequenze molto lunghe in modo più scalabile

🔹 Valutazione
Confronta i modelli con metriche di accuratezza
Analizza qualitativamente errori sistematici, ad esempio:
“il modello predice sempre la fine dell’azione con 5 frame di ritardo”

- Obiettivi 

🔸 Soft-NMS (Non-Maximum Suppression)
Post-processing per unire predizioni temporali sovrapposte
Riduce duplicati e migliora i confini delle azioni

🔸 Moduli basati su Mamba
Usa modelli di tipo state-space (alternativa a RNN/attention)
Obiettivo: migliorare la gestione di sequenze lunghe senza perdita di informazione
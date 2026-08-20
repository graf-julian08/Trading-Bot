# Algorithmic Trading Bot System

## Übersicht
Der **Trading-Bot** ist ein automatisierte Handelssystem in Python mit integriertem Risikomanagement, WebSocket-Marktdatenverarbeitung und maschinellen Lernmodellen.

## Projektstruktur & Architektur
- `main.py`: Haupteinstiegspunkt des Handelssystems.
- `data_engine.py` & `ws_manager.py`: Live-Datenverarbeitung und WebSocket-Verbindung.
- `execution_engine.py` & `risk_manager.py`: Orderausführung und Stop-Loss-Überwachung.
- `ai_model.py` & `train.py`: Modellarchitektur und Trainingsskripte.
- `optimizer.py` & `state_manager.py`: Systemoptimierung und Zustandsverwaltung.
- `tests_phase1.py` bis `tests_phase4.py`: Testsuite für Systemkomponenten.

## Hauptfunktionalitäten
- **Automatischer Handel**: Regelbasierte Ausführung von Aufträgen.
- **Risikomanagement**: Überwachung von Expositionsgrenzen und Stopp-Marken.
- **Echtzeit-Daten**: WebSocket-Verarbeitung von Börsen-Datenströmen.

## Ausführung & Nutzung
Nach Konfiguration der Parameter in `config.py` erfolgt der Start des Handelssystems über `python main.py`.

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz.

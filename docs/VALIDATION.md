# Validation

Pour chaque scénario : ce qui est **couvert automatiquement** (et par quel test) et ce qui reste à
**vérifier manuellement** sur votre machine. Tests automatiques : sources et détecteur **simulés**.

Légende : ✅ automatique · 🖐 manuel obligatoire · ✅+🖐 automatique (logique) + manuel (matériel réel)

| # | Scénario | Statut | Test automatique / procédure manuelle |
|---|---|---|---|
| 1 | Démarrage sans caméra : l'application reste ouverte, message clair | ✅ | `test_ui_smoke::test_startup_without_camera_stays_open_with_clear_message` (indicateur « Caméra : aucune détectée ») |
| 2 | Webcam locale : la vidéo s'affiche | 🖐 | Webcam locale → ⟳ → ▶ Démarrer. (Le rendu est couvert par `test_full_pipeline_display_table_and_stats` avec une source simulée.) |
| 3 | YOLO : objets détectés et annotés | 🖐 | `python tools/smoke_yolo.py photo.jpg`, puis l'application. (Parsing couvert par `test_parse_yolo_result_coordinates_and_labels`.) |
| 4 | Seuil modifié : résultats différents sans rechargement | ✅ | `test_confidence_changes_without_reload`, `test_worker_threshold_change_applies_live`, `test_full_pipeline_display_table_and_stats` (une seule instance de détecteur, `load_count == 1`) |
| 5 | Coordonnées dans le repère de l'image source | ✅+🖐 | Bornage et conversion : tests de parsing YOLO / Faster R-CNN. Réel : `smoke_yolo.py` affiche « OK » ou « HORS IMAGE » pour chaque boîte |
| 6 | Perte de connexion : erreur signalée, pas de gel | ✅ | `test_worker_streams_frames_then_signal_lost_and_reconnects`, `test_signal_loss_then_reconnection` |
| 7 | Reconnexion | ✅+🖐 | Mêmes tests (délai progressif, abandon après N tentatives : `test_worker_gives_up_after_max_attempts`). Réel : couper puis rétablir le Wi-Fi du téléphone |
| 8 | Changement de caméra : ancienne ressource libérée | ✅ | `test_manager_switch_source_releases_previous`, `test_changing_source_while_running_frees_previous_first` |
| 9 | Arrêt : workers arrêtés, caméra libérée | ✅ | `test_stop_releases_camera_and_threads`, `test_manager_start_stop_releases_source` |
| 10 | Fermeture de la fenêtre : aucun thread résiduel | ✅ | `test_shutdown_while_running_leaves_no_threads` |
| 11 | Charge élevée : interface réactive, files bornées | ✅+🖐 | `test_queue_stays_bounded_under_load`, `test_latest_queue_is_bounded_and_drops_oldest`. Réel : modèle lent + `frame_skip`, vérifier que la fenêtre reste fluide |
| 12 | Sans GPU : fonctionnement CPU | ✅+🖐 | `test_resolve_device_explicit_and_fallback`. Réel : machine sans GPU → barre supérieure « Modèle : … (cpu) » |

## Autres comportements couverts

- Modèle en échec : l'application reste ouverte, la vidéo s'affiche, aucune détection inventée
  (`test_model_failure_keeps_app_open`, `test_worker_inference_error_gives_no_fake_results_and_one_event`).
- Échec d'un changement de modèle : l'ancien modèle reste actif (`test_failed_switch_keeps_previous_model`).
- Source non prise en charge (Bluetooth, iOS direct) : erreur explicite, aucune reconnexion
  (`test_unsupported_source_does_not_retry`).
- URL invalide refusée avant tout démarrage (`test_invalid_url_is_rejected_without_starting`).
- Configuration corrompue ou hors bornes (`test_config_manager.py`).
- Chemins en mode exécutable gelé (`test_app_dir_normal_and_frozen`).

## Procédure de validation sur votre machine

```bash
python -m pytest                       # logique complète (l'interface s'exécute si un affichage est disponible)
python tools/smoke_yolo.py photo.jpg   # YOLO réel : chargement, 3 seuils, coordonnées
python main.py                         # scénarios 2, 3, 7, 11, 12 à la main
```

Ce qui n'a **pas** été exécuté par l'auteur : YOLO / Faster R-CNN réels, webcam physique, Windows, Python 3.14,
build PyInstaller avec PyTorch.

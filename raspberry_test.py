import os
import cv2
import time
import numpy as np
import pandas as pd
import requests
from ultralytics import YOLO
from datetime import datetime
import mimetypes
import json
import RPi.GPIO as GPIO

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")  # Veillez à ce que ce modèle soit bien disponible

# Créer le dossier de sortie s’il n’existe pas
os.makedirs("output", exist_ok=True)

# Configuration du buzzer (GPIO 17)
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def detect_bird_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    dict_results = {
        'frame_idx' : [],
        'birds_in_frame' : [],
        'prob_min' : [],
        'prob_max' : [],
        'prob_avg' : []
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        classes = results.boxes.cls
        scores = results.boxes.conf
        bird_scores = [float(scores[i]) for i in range(len(classes)) if int(classes[i]) == 14]
        birds_in_frame = len(bird_scores)

        if birds_in_frame > 0:
            prob_min = min(bird_scores)
            prob_max = max(bird_scores)
            prob_avg = sum(bird_scores) / birds_in_frame
        else:
            prob_min = prob_max = prob_avg = 0.0

        dict_results['frame_idx'].append(frame_idx)
        dict_results['birds_in_frame'].append(birds_in_frame)
        dict_results['prob_min'].append(prob_min)
        dict_results['prob_max'].append(prob_max)
        dict_results['prob_avg'].append(prob_avg)

        annotated_frame = results.plot()
        cv2.imwrite(os.path.join("output", f"frame_{frame_idx:04d}.jpg"), annotated_frame)
        frame_idx += 1

    cap.release()

    df_result = pd.DataFrame(dict_results)
    output = df_result[df_result['birds_in_frame']==df_result['birds_in_frame'].max()].to_dict('records')[0]
    return output

def send_alert(video_path, media_path):
    url = "https://server-agriproject.onrender.com/api/detections"
    if not os.path.exists(media_path):
        print(f"Erreur: Le fichier {media_path} n'existe pas")
        return
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier {video_path} n'existe pas")
        return
    media_mime = mimetypes.guess_type(media_path)[0] or 'image/jpeg'
    tram_mime = mimetypes.guess_type(video_path)[0] or 'video/mp4'
    files = {
        'media': ('media', open(media_path, 'rb'), media_mime),
        'tram': ('tram', open(video_path, 'rb'), tram_mime)
    }
    data = {'systeme_id': 2}
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Python/Requests',
        'Connection': 'keep-alive'
    }
    session = requests.Session()
    session.max_redirects = 5
    try:
        response = session.post(
            url,
            files=files,
            data=data,
            headers=headers,
            timeout=120,
            allow_redirects=True
        )
        print(f"Code de statut: {response.status_code}")
        if response.status_code == 201:
            print("Succès! Détection créée avec succès")
            print("Réponse:", response.json())
        else:
            print(f"Erreur: {response.status_code}")
            try:
                print("Détails:", response.json())
            except json.JSONDecodeError:
                print("Contenu brut de la réponse:", response.text)
    except Exception as e:
        print(f"Erreur lors de la requête: {e}")
    finally:
        files['media'][1].close()
        files['tram'][1].close()
        session.close()

def main():
    video_file = "vid.mp4"
    print("[INFO] Utilisation de la vidéo existante :", video_file)
    result = detect_bird_in_video(video_file)
    print("[INFO] Résultat :", result)
    if result['birds_in_frame'] > 6:
        print("[ALERTE] Trop d'oiseaux détectés. Activation du buzzer.")
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        send_alert(video_file, f"output/frame_{result['frame_idx']:04d}.jpg")
    else:
        print("[INFO] Nombre d'oiseaux détectés acceptable. Pas d'alerte envoyée.")

if __name__ == "__main__":
    try:
        main()
    finally:
        GPIO.cleanup() 
    
# **Instructions pour lancer les conteneurs**

Ce projet utilise Docker pour exécuter deux services principaux :
- **`serving-api`** : Une API FastAPI pour prédire les émotions à partir de fichiers audio.
- **`webapp`** : Une interface utilisateur Streamlit pour interagir avec l'API.

## **Prérequis**
1. Docker et Docker Compose doivent être installés sur votre machine.
2. Assurez-vous que le réseau Docker `shared_net` est créé :
   ```bash
   docker network create shared_net
   ```

---

## **Étapes pour démarrer les services**

### **1. Lancer le service `serving-api`**
1. Allez dans le répertoire `serving` :
   ```bash
   cd serving
   ```
2. Lancez le service `serving-api` :
   ```bash
   docker-compose up --build
   ```
3. Vérifiez que l'API est accessible à l'adresse suivante : [http://localhost:8080/docs](http://localhost:8080/docs).

---

### **2. Lancer le service `webapp`**
1. Allez dans le répertoire `webapp` :
   ```bash
   cd webapp
   ```
2. Connectez manuellement `serving-api` au réseau `shared_net` (si nécessaire) :
   ```bash
   docker network connect shared_net serving-api
   ```
3. Lancez le service `webapp` :
   ```bash
   docker-compose up --build
   ```
4. Vérifiez que l'interface utilisateur est accessible à l'adresse suivante : [http://localhost:8081](http://localhost:8081).

---

## **Dépannage**

### **Problème : `webapp` ne peut pas se connecter à `serving-api`**
1. Assurez-vous que les deux services sont connectés au réseau `shared_net` :
   ```bash
   docker network inspect shared_net
   ```
   Les conteneurs `serving-api` et `webapp` doivent être listés dans la section `Containers`.

2. Reconnectez les conteneurs au réseau si nécessaire :
   ```bash
   docker network connect shared_net serving-api
   docker network connect shared_net webapp
   ```

3. Relancez les services.

---

## **Arrêter les services**
Pour arrêter les services, exécutez les commandes suivantes dans les dossiers respectifs :
1. Arrêter `serving-api` :
   ```bash
   docker-compose down
   ```
2. Arrêter `webapp` :
   ```bash
   docker-compose down
   ```

---

Avec ces instructions, vous pourrez configurer et exécuter les conteneurs correctement. Si vous rencontrez des problèmes, n'hésitez pas à demander de l'aide.

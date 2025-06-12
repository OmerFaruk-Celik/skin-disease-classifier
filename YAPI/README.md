Project Development Workflow
This document outlines the development workflow, including Git branch strategies, their corresponding Jira tasks, descriptions, and Git commands used throughout the project. Each branch represents a specific feature or release, aligned with Jira epics and tasks.
Branch: feature/TASK-2-ui-scaffolding
Jira Correspondence:

Epic: User Interface (UI) Development
Tasks: TASK-1, TASK-2 (Creation of main screen and basic page structures)

Description:
In the initial phase, we created a feature branch from main for TASK-2. This branch focused on developing the visual scaffolding for the Login, Register, and Home screens using Flutter. Upon completion, the branch was merged back into main, and the project was tagged with version v0.1.0.
Git Commands:
# 1. Create a new feature branch from main
git checkout main
git checkout -b feature/TASK-2-ui-scaffolding

# 2. Add Flutter UI files (Login.dart, Register.dart, etc.)
git add lib/Screens/
git commit -m "feat(ui): TASK-2 - Login ve Register ekranlarının temel UI'ı oluşturuldu"

# 3. Merge back to main
git checkout main
git merge --no-ff feature/TASK-2-ui-scaffolding

# 4. Tag the milestone
git tag -a v0.1.0 -m "Sürüm 0.1.0: Temel UI ve Kimlik Doğrulama Ekranları"

Branch: feature/TASK-5-backend-and-initial-model
Jira Correspondence:

Epic: Database and Backend Infrastructure / AI Model Integration
Tasks: TASK-3, TASK-4, TASK-5

Description:
This branch was created for TASK-5 to establish the backend infrastructure. We developed server.py using Flask, integrated Firebase, and incorporated the first trained AI model into the API, enabling basic predictions. The changes were merged into main and tagged as version v0.2.0.
Git Commands:
# 1. Create a new branch for backend setup
git checkout -b feature/TASK-5-backend-and-initial-model

# 2. Add backend files (server.py, train.py, model.tflite) and Firebase config
git add backend/ lib/firebase_options.dart
git commit -m "feat(backend,ai): TASK-5 - Flask API ve ilk AI modeli entegre edildi"

# 3. Merge to main
git checkout main
git merge --no-ff feature/TASK-5-backend-and-initial-model

# 4. Tag the release
git tag -a v0.2.0 -m "Sürüm 0.2.0: Çalışan Backend API ve İlk AI Modeli"

Branch: feature/TASK-11-model-improvement
Jira Correspondence:

Epic: AI Model Integration
Task: TASK-11 (Model improvement with data balancing)

Description:
This branch was dedicated to enhancing the AI model's performance for TASK-11. We updated the train.py script with data augmentation and class weighting techniques, improving model accuracy through refactoring. The changes were merged into main and tagged as v0.3.0.
Git Commands:
# 1. Create a branch for model improvement
git checkout -b feature/TASK-11-model-improvement

# 2. Update train.py
git add backend/train.py
git commit -m "refactor(ai): TASK-11 - Model eğitimi veri dengeleme ile iyileştirildi"

# 3. Merge to main
git checkout main
git merge --no-ff feature/TASK-11-model-improvement

# 4. Tag the release
git tag -a v0.3.0 -m "Sürüm 0.3.0: Geliştirilmiş Veri İşleme ile AI Modeli"

Branch: feature/TASK-6-camera-module
Jira Correspondence:

Epic: Image Processing and Input Collection
Task: TASK-6 (Capturing visual data with camera module)

Description:
In this branch, we connected the mobile app to the backend for TASK-6. Using the image_picker package in Flutter, we implemented camera and gallery access. Functions were developed to send user-selected photos to the backend API via HTTP POST requests. The branch was merged into main and tagged as v0.4.0.
Git Commands:
# 1. Create a branch for camera module
git checkout -b feature/TASK-6-camera-module

# 2. Add/update Camera.dart, Galeri.dart, and HTTP functions
git add lib/
git commit -m "feat(mobile): TASK-6 - Kamera ve galeri ile görüntü yükleme özelliği eklendi"

# 3. Merge to main
git checkout main
git merge --no-ff feature/TASK-6-camera-module

# 4. Tag the release
git tag -a v0.4.0 -m "Sürüm 0.4.0: Mobil Görüntü Yükleme Fonksiyonu"

Branch: feature/TASK-12-upgrade-to-mobilenetv3
Jira Correspondence:

Epic: AI Model Integration
Task: TASK-12 (Upgrading model architecture to MobileNetV3)

Description:
To improve results, we upgraded the model architecture to MobileNetV3Large in this branch for TASK-12. The train.py script was updated to utilize this modern, efficient architecture. After completion, the branch was merged into main and tagged as v0.5.0.
Git Commands:
# 1. Create a branch for model architecture upgrade
git checkout -b feature/TASK-12-upgrade-to-mobilenetv3

# 2. Update train.py for MobileNetV3
git add backend/train.py
git commit -m "feat(ai): TASK-12 - Model mimarisi MobileNetV3'e geçirildi"

# 3. Merge to main
git checkout main
git merge --no-ff feature/TASK-12-upgrade-to-mobilenetv3

# 4. Tag the release
git tag -a v0.5.0 -m "Sürüm 0.5.0: AI Modeli MobileNetV3'e Yükseltildi"

Branch: release/v1.0.0
Jira Correspondence:

Epic: User Feedback / Testing and Debugging
Tasks: TASK-8, TASK-9, TASK-15, TASK-16

Description:
For the final phase, we created a release branch to prepare the first stable version (v1.0.0). We enhanced the backend API with error handling and logging, finalized all Flutter app screens and user flows, and updated the project documentation in README.md. After final touches, the branch was merged into main, and the project was tagged as v1.0.0.
Git Commands:
# 1. Create a release branch
git checkout -b release/v1.0.0

# 2. Add/update all final files (Flutter, Backend, README)
git add .
git commit -m "feat!: Projenin ilk stabil sürümü için tüm bileşenler tamamlandı"

# 3. Merge to main
git checkout main
git merge --no-ff release/v1.0.0

# 4. Tag the final release
git tag -a v1.0.0 -m "Sürüm 1.0.0: İlk Stabil Sürüm"


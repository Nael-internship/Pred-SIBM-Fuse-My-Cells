FuseMyCells 🔬+🤖 = 2×🔬

Ma participation au challenge France-BioImaging

Voici mon dépôt dédié au challenge Fuse My Cells organisé par France-BioImaging.
J’y rassemble mon travail, mes essais, mes choix méthodologiques et les scripts que j’ai développés pour préparer les données et évaluer mes méthodes.

Pour plus d’informations sur le challenge :

fusemycells.grand-challenge.org

Annonce France-BioImaging

📂 Usage
Préparation du dataset

C’est la procédure que j’ai suivie pour préparer les données :

Télécharger tous les fichiers .zip et les placer dans un même dossier.
Les instructions suivantes doivent être exécutées depuis ce dossier.

Lancer 01_unzip.py
Les scripts liés à la préparation des données se trouvent dans le dossier data/.

Lancer 02_tif_to_hdf5.py

À l’issue de ces étapes, toutes les images extraites se trouvent dans un dossier images/, puis sont regroupées dans un unique fichier FuseMyCells.hdf5.

Lancer une évaluation
usage: eval.py [-h] [--use-gpu] --method {gaussian_filter,denoise_wavelet,denoise_tv_bregman} [--args ARGS [ARGS ...]]
               [--dataset DATASET] [--crop-data]
eval.py: error: the following arguments are required: --method
Exemple d’appel que j’utilise :
python eval.py --method gaussian_filter --args sigma=0.5 --dataset FuseMyCells.hdf5

Je suis parti du docker_template
 fourni par l’organisation, que j’ai adapté à ma démarche.
 from scipy import ndimage
if metadata['channel'] == 'nucleus':
    image_predict = ndimage.gaussian_filter(image_input, 0.442)
else:
    image_predict = ndimage.gaussian_filter(image_input, 0.5)

Les valeurs de sigma ont été choisies manuellement après plusieurs évaluations sur le dataset d’entraînement.
L’évaluation s’effectue via eval.py, et dans mon cas j’utilise plus souvent le script run.sh pour automatiser mes tests.

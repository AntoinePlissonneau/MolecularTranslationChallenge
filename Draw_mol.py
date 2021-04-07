# ====================================================
# Library
# ====================================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

# ====================================================
# Data Loading
# ====================================================
df = pd.read_csv(r"F://bms-molecular-translation//train_labels.csv")

# ====================================================
# Drawing
# ====================================================
for _, row in df.head(5).iterrows():
    ## read original image from dataset
    img_id = row["image_id"]
    img = cv2.imread("F://bms-molecular-translation//train//{}//{}//{}//{}.png".format(img_id[0], img_id[1], img_id[2], img_id), cv2.IMREAD_COLOR)

    ## read International Chemical Identifier
    mol = Chem.inchi.MolFromInchi(row["InChI"])

    ## draw molecule with 0 degree rotation
    d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(512, 512)
    AllChem.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.drawOptions().useBWAtomPalette()
    #d.drawOptions().rotate = 0
    #d.drawOptions().bondLineWidth = 1
    d.WriteDrawingText("0.png")
    img0 = cv2.imread("0.png", cv2.IMREAD_COLOR)

    ## draw molecule with 90 degree rotation
    d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(512, 512)
    AllChem.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().rotate = 90
    # d.drawOptions().bondLineWidth = 1
    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.WriteDrawingText("90.png")
    img90 = cv2.imread("90.png", cv2.IMREAD_COLOR)

    ## draw molecule with 180 degree rotation
    d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(512, 512)
    AllChem.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().rotate = 180
    # d.drawOptions().bondLineWidth = 1
    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.WriteDrawingText("180.png")
    img180 = cv2.imread("180.png", cv2.IMREAD_COLOR)

    # show images
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 4, 1), plt.imshow(img)
    plt.subplot(1, 4, 2), plt.imshow(img0)
    plt.subplot(1, 4, 3), plt.imshow(img90)
    plt.subplot(1, 4, 4), plt.imshow(img180)
    plt.show()
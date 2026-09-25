import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Composants OCR chargés à la demande.
# Aucun modèle EasyOCR n'est initialisé lors de l'import du module.
_ocr_reader = None
_ocr_pymupdf = None
_ocr_image_class = None
_ocr_initialization_attempted = False


def _get_ocr_components():
    """
    Charge PyMuPDF, Pillow et EasyOCR uniquement lorsqu'un fallback OCR
    est réellement nécessaire.

    Le chargement est effectué au maximum une fois par processus.
    """
    global _ocr_reader
    global _ocr_pymupdf
    global _ocr_image_class
    global _ocr_initialization_attempted

    if _ocr_initialization_attempted:
        return (
            _ocr_pymupdf,
            _ocr_image_class,
            _ocr_reader,
        )

    _ocr_initialization_attempted = True

    try:
        import pymupdf
        from PIL import Image
        import easyocr

        logging.info("Initialisation du lecteur EasyOCR...")

        _ocr_reader = easyocr.Reader(["en", "fr"])
        _ocr_pymupdf = pymupdf
        _ocr_image_class = Image

        logging.info("Lecteur EasyOCR initialisé.")

    except ImportError as error:
        logging.warning(
            "Dépendances OCR indisponibles : %s. "
            "Le fallback OCR sera désactivé.",
            error,
        )

    except Exception as error:
        logging.error(
            "Impossible d'initialiser EasyOCR : %s",
            error,
        )

    return (
        _ocr_pymupdf,
        _ocr_image_class,
        _ocr_reader,
    )


def extract_text_from_pdf_with_ocr(
    file_path: str,
) -> Optional[str]:
    """
    Extrait le texte d'un PDF avec EasyOCR.

    Les dépendances OCR et le modèle ne sont chargés qu'au premier
    appel effectif de cette fonction.
    """
    pymupdf, image_class, ocr_reader = _get_ocr_components()

    if (
        pymupdf is None
        or image_class is None
        or ocr_reader is None
    ):
        logging.warning(
            "OCR indisponible pour %s.",
            file_path,
        )
        return None

    text_content = []

    try:
        document = pymupdf.open(file_path)

        for page_num in tqdm(
            range(len(document)),
            desc=f"OCR de {os.path.basename(file_path)}",
        ):
            page = document.load_page(page_num)

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2)
            )

            image = image_class.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            try:
                image_array = np.array(image)
                results = ocr_reader.readtext(image_array)

                page_text = "\n".join(
                    result[1]
                    for result in results
                )

                text_content.append(page_text)

            except Exception as error:
                logging.error(
                    "Erreur OCR page %s de %s : %s",
                    page_num + 1,
                    file_path,
                    error,
                )

        document.close()

        full_text = "\n".join(text_content).strip()

        if full_text:
            logging.info(
                "Texte extrait via OCR de %s (%s caractères).",
                file_path,
                len(full_text),
            )
            return full_text

        logging.warning(
            "Aucun texte significatif extrait via OCR de %s.",
            file_path,
        )
        return None

    except Exception as error:
        logging.error(
            "Erreur pendant le traitement OCR de %s : %s",
            file_path,
            error,
        )
        return None


def extract_text_from_pdf(
    file_path: str,
) -> Optional[str]:
    """
    Extrait le texte d'un PDF avec PyPDF2.

    EasyOCR n'est utilisé qu'en fallback si l'extraction standard
    produit moins de 100 caractères ou échoue.
    """
    try:
        from PyPDF2 import PdfReader

        pdf_reader = PdfReader(file_path)

        extracted_pages = []

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                extracted_pages.append(page_text)

        text = "\n".join(extracted_pages)

        if len(text.strip()) < 100:
            logging.info(
                "Peu de texte trouvé dans %s via extraction standard "
                "(%s caractères). Tentative d'OCR...",
                file_path,
                len(text.strip()),
            )

            ocr_text = extract_text_from_pdf_with_ocr(file_path)

            if ocr_text:
                return ocr_text

            logging.warning(
                "Le fallback OCR n'a pas produit de texte "
                "significatif pour %s.",
                file_path,
            )
            return text

        logging.info(
            "Texte extrait de PDF : %s (%s caractères).",
            file_path,
            len(text),
        )
        return text

    except Exception as error:
        logging.error(
            "Erreur extraction PDF %s : %s. "
            "Tentative d'OCR en dernier recours...",
            file_path,
            error,
        )

        return extract_text_from_pdf_with_ocr(file_path)


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info(f"Texte extrait de DOCX: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction DOCX {file_path}: {e}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logging.info(f"Texte extrait de TXT: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction TXT {file_path}: {e}")
        return None

def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV (convertit en string)."""
    try:
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1') # Essayer un autre encodage courant
        except Exception as read_e:
             logging.warning(f"Erreur lecture CSV {file_path}: {read_e}. Tentative avec séparateur ';'")
             try:
                 df = pd.read_csv(file_path, sep=';')
             except UnicodeDecodeError:
                  df = pd.read_csv(file_path, sep=';', encoding='latin1')
             except Exception as read_e2:
                   logging.error(f"Impossible de lire le CSV {file_path}: {read_e2}")
                   return None

        text = df.to_string()
        logging.info(f"Texte extrait de CSV: {file_path} ({len(text)} caractères)")
        return text
    except ImportError:
        logging.warning("Pandas non installé. Impossible de lire les fichiers CSV.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction CSV {file_path}: {e}")
        return None

def extract_text_from_excel(file_path: str) -> Optional[Union[str, Dict[str, str]]]:
    """Extrait le texte de chaque feuille d'un fichier Excel."""
    try:
        import pandas as pd
        # Lire toutes les feuilles dans un dictionnaire de DataFrames
        excel_file = pd.ExcelFile(file_path)
        sheets_data = {}
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            sheets_data[sheet_name] = df.to_string()
        
        logging.info(f"Texte extrait de {len(sheets_data)} feuille(s) dans Excel: {file_path}")
        # Si une seule feuille, retourne directement le texte pour la compatibilité
        if len(sheets_data) == 1:
            return list(sheets_data.values())[0]
        return sheets_data
    except ImportError:
        logging.warning("Pandas ou openpyxl non installé. Impossible de lire les fichiers Excel.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction Excel {file_path}: {e}")
        return None

# --- Fonctions de chargement ---

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un fichier ZIP depuis une URL et l'extrait."""
    if not url:
        logging.warning("Aucune URL fournie pour le téléchargement.")
        return False
    try:
        logging.info(f"Téléchargement des données depuis {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info(f"Extraction du contenu dans {output_dir}...")
            z.extractall(output_dir)
        logging.info("Téléchargement et extraction terminés.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de téléchargement: {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("Le fichier téléchargé n'est pas un ZIP valide.")
        return False
    except Exception as e:
        logging.error(f"Erreur inattendue lors du téléchargement/extraction: {e}")
        return False

def load_and_parse_files(input_dir: str) -> List[Dict[str, any]]:
    """
    Charge et parse récursivement les fichiers d'un répertoire.
    Retourne une liste de dictionnaires, chacun représentant un document.
    """
    documents = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return []

    logging.info(f"Parcours du répertoire source: {input_dir}")
    for file_path in input_path.rglob("*.*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(input_path)
            source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
            ext = file_path.suffix.lower()
            
            logging.debug(f"Traitement du fichier: {relative_path} (Dossier source: {source_folder})")

            extracted_content = None
            if ext == ".pdf":
                extracted_content = extract_text_from_pdf(str(file_path))
            elif ext == ".docx":
                extracted_content = extract_text_from_docx(str(file_path))
            elif ext == ".txt":
                extracted_content = extract_text_from_txt(str(file_path))
            elif ext == ".csv":
                extracted_content = extract_text_from_csv(str(file_path))
            elif ext in [".xlsx", ".xls"]:
                extracted_content = extract_text_from_excel(str(file_path))
            # Suppression de la gestion des fichiers HTML
            else:
                logging.warning(f"Type de fichier non supporté ignoré: {relative_path}")
                continue

            if not extracted_content:
                logging.warning(f"Aucun contenu n'a pu être extrait de {relative_path}")
                continue
            
            # Si c'est un dictionnaire (plusieurs feuilles Excel), créer un doc par feuille
            if isinstance(extracted_content, dict):
                for sheet_name, text in extracted_content.items():
                    documents.append({
                        "page_content": text,
                        "metadata": {
                            "source": f"{str(relative_path)} (Feuille: {sheet_name})",
                            "filename": file_path.name,
                            "sheet": sheet_name,
                            "category": source_folder,
                            "relative_path": str(relative_path)
                        }
                    })
            else: # Pour tous les autres types de fichiers
                 documents.append({
                    "page_content": extracted_content,
                    "metadata": {
                        "source": str(relative_path),
                        "filename": file_path.name,
                        "category": source_folder,
                        "relative_path": str(relative_path)
                    }
                })

    logging.info(f"{len(documents)} documents chargés et parsés.")
    return documents
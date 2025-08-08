from pathlib import Path
import subprocess, shutil


class DocumentConverter:
    def __init__(self, out_dir:Path):
        self.out_dir = out_dir

        self.soffice = shutil.which("soffice")
        if not self.soffice:
            raise RuntimeError("LibreOffice (soffice) non trouvé dans le PATH")  

    def to_pdf(self, doc_path: Path) -> Path:
        """
        Convertit un .doc/.docx en PDF via LibreOffice headless,
        et place le PDF dans `out_dir` (par défaut PDF_OUTPUT_DIR).
        """

        # assurez-vous que le dossier de sortie existe
        self.out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.soffice,
            "--headless",
            "--invisible",
            "--convert-to", "pdf",
            "--outdir", str(self.out_dir),  # Spécifie explicitement le dossier de sortie
            str(doc_path),
        ]
        
        # Capture la sortie pour le débogage
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERREUR: {result.stderr}")
            raise RuntimeError(f"LibreOffice a échoué avec le code {result.returncode}")

        pdf_path = self.out_dir / f"{doc_path.stem}.pdf"
        if not pdf_path.exists():
            print(f"Recherche de: {pdf_path}")
            print(f"Contenu du dossier: {list(self.out_dir.iterdir())}")
            raise FileNotFoundError(f"Échec de la conversion : {pdf_path} introuvable.")
        
        print(f"✅ PDF créé: {pdf_path}")
        return pdf_path
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import torch
from tqdm import tqdm
import re
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing numbers, punctuation, and extra spaces."""
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

class AcademicianProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.academic_texts: List[str] = []
        self.academic_sources: List[Tuple[str, str, str]] = []
        self.cleaner = TextCleaner()

    def load_academicians(self) -> None:
        """Load and process academician data from JSON file."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                academicians = json.load(f)
            
            for academician in academicians:
                self._process_academician(academician)
                
        except FileNotFoundError:
            logger.error(f"Academicians file not found: {self.file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {self.file_path}")
            raise

    def _process_academician(self, academician: Dict[str, Any]) -> None:
        """Process a single academician's data."""
        fullname = academician.get("Fullname", "")
        
        # Process duties
        for duty in academician.get("Duties", []):
            text = self.cleaner.clean_text(f"{duty.get('Year', '')} {duty.get('Title', '')} {duty.get('University', '')}")
            self.academic_texts.append(text)
            self.academic_sources.append((fullname, "Görev", text))
            
        # Process books
        for book in academician.get("Books", []):
            text = self.cleaner.clean_text(f"{book.get('Title', '')} {book.get('Year', '')} {book.get('Description', '')}")
            self.academic_texts.append(text)
            self.academic_sources.append((fullname, "Kitap", text))
            
        # Process articles
        for article in academician.get("Articles", []):
            text = self.cleaner.clean_text(f"{article.get('Title', '')} {article.get('Description', '')}")
            self.academic_texts.append(text)
            self.academic_sources.append((fullname, "Makale", text))
            
        # Process declarations
        for declaration in academician.get("Declarations", []):
            text = self.cleaner.clean_text(f"{declaration.get('Title', '')} {declaration.get('Description', '')}")
            self.academic_texts.append(text)
            self.academic_sources.append((fullname, "Bildiri", text))
            
        # Process projects
        for project in academician.get("Projects", []):
            text = self.cleaner.clean_text(f"{project.get('Title', '')} {project.get('Description', '')}")
            self.academic_texts.append(text)
            self.academic_sources.append((fullname, "Proje", text))
            
        # Process thesis
        for thesis in academician.get("Thesis", []):
            text = self.cleaner.clean_text(f"{thesis.get('Period', '')} {thesis.get('CreatorName', '')} {thesis.get('ThesisName', '')} {thesis.get('University', '')}")
            self.academic_texts.append(text)
            self.academic_sources.append((fullname, "Tez", text))

class YazarBazliProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.yazar_texts: List[str] = []
        self.yazar_sources: List[Tuple[str, str, str]] = []
        self.cleaner = TextCleaner()

    def load_yazar_kayitlar(self) -> None:
        """Load and process yazar bazlı kayıtlar from JSON file."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                yazar_kayitlar = json.load(f)
            
            # Process each author's publications
            for yazar, publications in yazar_kayitlar.items():
                for publication in publications:
                    # Combine title and abstract for better similarity matching
                    text = self.cleaner.clean_text(f"{publication.get('Article Title', '')} {publication.get('Abstract', '')}")
                    self.yazar_texts.append(text)
                    self.yazar_sources.append((yazar, "Makale", text))
                    
        except FileNotFoundError:
            logger.error(f"Yazar bazlı kayıtlar file not found: {self.file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {self.file_path}")
            raise

class ProjectProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.project_texts: List[str] = []
        self.project_df = None
        self.cleaner = TextCleaner()

    def load_projects(self) -> None:
        """Load and process project data."""
        try:
            self.project_df = pd.read_csv(self.file_path)
            self.project_df["combined_text"] = (
                self.project_df["title"].fillna("") + " " + 
                self.project_df["objective_clean"].fillna("")
            ).apply(self.cleaner.clean_text)
            self.project_texts = self.project_df["combined_text"].tolist()
        except FileNotFoundError:
            logger.error(f"Projects file not found: {self.file_path}")
            raise

class SimilarityCalculator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        self.batch_size = 32

    def calculate_embeddings(self, texts: List[str], desc: str) -> torch.Tensor:
        """Calculate embeddings for a list of texts."""
        logger.info(f"Starting embedding calculation for {len(texts)} texts")
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        try:
            for i in tqdm(range(0, len(texts), self.batch_size), desc=desc):
                batch = texts[i:i+self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1}/{total_batches}")
                
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_tensor=True, 
                    show_progress_bar=False, 
                    device=self.device
                )
                embeddings.append(batch_embeddings)
                
                # Clear CUDA cache if using GPU
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            logger.info("Embedding calculation completed successfully")
            return torch.cat(embeddings)
            
        except Exception as e:
            logger.error(f"Error during embedding calculation: {str(e)}", exc_info=True)
            raise

    def find_similarities(
        self, 
        project_embeddings: torch.Tensor, 
        academic_embeddings: torch.Tensor,
        project_df: pd.DataFrame,
        academic_sources: List[Tuple[str, str, str]],
        threshold: float = 0.5,  # Decreased threshold to get more matches
        top_k: int = 5  # Only keep top k matches per project
    ) -> List[Dict[str, Any]]:
        """Find similarities between project and academic embeddings."""
        similarities = util.cos_sim(project_embeddings, academic_embeddings)
        results = []
        
        for i in tqdm(range(len(project_df)), desc="🔍 Benzerlik Eşleşmeleri"):
            project_row = project_df.iloc[i]
            # Get top k matches for this project
            project_similarities = similarities[i]
            top_indices = torch.topk(project_similarities, k=min(top_k, len(project_similarities)))[1]
            
            for j in top_indices:
                score = project_similarities[j].item()
                if score >= threshold:
                    fullname, source_type, source_text = academic_sources[j]
                    results.append({
                        "Project Title": project_row["title"],
                        "Match Type": source_type,
                        "Matched By": fullname,
                        "Similarity": round(score, 4)
                    })
        return results

def main():
    try:
        logger.info("Starting similarity calculation process...")
        
        # Initialize processors with correct file paths
        logger.info("Initializing processors...")
        academic_processor = AcademicianProcessor("scrape/Sbert/academicians.json")
        yazar_processor = YazarBazliProcessor("scrape/Sbert/yazar_bazli_kayitlar.json")
        project_processor = ProjectProcessor("scrape/Sbert/cleaned_projects.csv")
        
        # Load data
        logger.info("Loading academician data...")
        academic_processor.load_academicians()
        logger.info(f"Loaded {len(academic_processor.academic_texts)} academician records")
        
        logger.info("Loading yazar bazlı kayıtlar...")
        yazar_processor.load_yazar_kayitlar()
        logger.info(f"Loaded {len(yazar_processor.yazar_texts)} yazar records")
        
        logger.info("Loading project data...")
        project_processor.load_projects()
        logger.info(f"Loaded {len(project_processor.project_texts)} project records")
        
        # Initialize similarity calculator
        logger.info("Initializing similarity calculator...")
        calculator = SimilarityCalculator()
        logger.info(f"Using device: {calculator.device}")
        
        # Calculate embeddings
        logger.info("Calculating academic embeddings...")
        academic_embeddings = calculator.calculate_embeddings(academic_processor.academic_texts, "📚 Academic Embedding")
        logger.info("Academic embeddings calculated successfully")
        
        logger.info("Calculating yazar embeddings...")
        yazar_embeddings = calculator.calculate_embeddings(yazar_processor.yazar_texts, "📚 Yazar Embedding")
        logger.info("Yazar embeddings calculated successfully")
        
        logger.info("Calculating project embeddings...")
        project_embeddings = calculator.calculate_embeddings(project_processor.project_texts, "🛠 Project Embedding")
        logger.info("Project embeddings calculated successfully")
        
        # Find similarities
        logger.info("Finding similarities between projects and academic records...")
        similarities = util.cos_sim(project_embeddings, academic_embeddings)
        yazar_similarities = util.cos_sim(project_embeddings, yazar_embeddings)
        results = []
        
        for i in tqdm(range(len(project_processor.project_df)), desc="🔍 Benzerlik Eşleşmeleri"):
            project_row = project_processor.project_df.iloc[i]
            project_similarities = similarities[i]
            yazar_project_similarities = yazar_similarities[i]
            
            # Get top 5 matches for this project from academic records
            top_indices = torch.topk(project_similarities, k=min(5, len(project_similarities)))[1]
            for j in top_indices:
                score = project_similarities[j].item()
                if score >= 0.5:
                    fullname, source_type, source_text = academic_processor.academic_sources[j]
                    results.append({
                        "Akademisyen": fullname,
                        "EU Projesi": project_row["title"],
                        "EU ID": project_row["id"],
                        "Benzerlik Oranı": round(score, 4),
                        "Eşleşme Tipi": source_type
                    })
            
            # Get top 5 matches for this project from yazar records
            top_yazar_indices = torch.topk(yazar_project_similarities, k=min(5, len(yazar_project_similarities)))[1]
            for j in top_yazar_indices:
                score = yazar_project_similarities[j].item()
                if score >= 0.5:
                    yazar, source_type, source_text = yazar_processor.yazar_sources[j]
                    results.append({
                        "Akademisyen": yazar,
                        "EU Projesi": project_row["title"],
                        "EU ID": project_row["id"],
                        "Benzerlik Oranı": round(score, 4),
                        "Eşleşme Tipi": source_type
                    })
        
        logger.info(f"Found {len(results)} similarity matches")
        
        # Save results with optimized data types
        logger.info("Saving results to CSV...")
        results_df = pd.DataFrame(results)
        # Sort by similarity score in descending order
        results_df = results_df.sort_values("Benzerlik Oranı", ascending=False)
        
        # Optimize data types
        results_df["Benzerlik Oranı"] = results_df["Benzerlik Oranı"].astype("float32")
        results_df["Akademisyen"] = results_df["Akademisyen"].astype("string")
        results_df["EU Projesi"] = results_df["EU Projesi"].astype("string")
        results_df["EU ID"] = results_df["EU ID"].astype("string")
        results_df["Eşleşme Tipi"] = results_df["Eşleşme Tipi"].astype("string")
        
        output_path = "scrape/Sbert/sbert_test_results.csv"
        # Save with UTF-8 encoding and BOM to handle Turkish characters correctly
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ Test completed. Results written to '{output_path}'")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
# Assuming this is your import based on the original code
# from docling.document_converter import DocumentConverter
import shutil
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Tuple, List
import torch
from docling_core.types import DoclingDocument
from tqdm import tqdm

from src.preprocessing.corrupted_docx_analyzer import DocxXMLAttributeAnalyzer
from src.preprocessing.docx_graphics_datafixer import DocxGraphicsDataFixer
from src.utils.file_manager import scan_folder, sanitize_filename
from docling.document_converter import DocumentConverter


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self, input_folder_path: str, output_folder_path: str):
        self.input_folder_path = Path(input_folder_path)
        self.output_folder_path = Path(output_folder_path)

    @property
    @abstractmethod
    def extension(self) -> str:
        """
        The file extension this parser supports (e.g., '.pdf', '.docx').

        Returns:
            str: File extension (including the dot, e.g., '.pdf').
        """
        pass

    @property
    def output_extension(self) -> str:
        return ".json"

    @property
    def __corrupted_docx_folder(self) -> Path:
        return self.output_folder_path / "corrupted_docx"

    @property
    def _parsed_docx_folder(self) -> Path:
        return self.output_folder_path / "parsed_docx"

    def __process_corrupted_file(self, file_path: Path) -> tuple[bool, Path] | tuple[bool, None]:
        try:
            # create directory in the following path if not exist
            skipped_path = self.__corrupted_docx_folder / sanitize_filename(file_path.stem)
            skipped_path.mkdir(parents=True, exist_ok=True)
            destination = skipped_path / file_path.name
            shutil.copy2(file_path, destination)
            # Analyze the file
            analyzer = DocxXMLAttributeAnalyzer(max_length_threshold=1000000)
            results = analyzer.analyze_docx_file(docx_path=destination.as_posix())
            # print(analyzer.generate_report(include_context=True))
            if results:
                print("Found problems in DOCx related to gfxdata included data")
                analyzer.export_problems_to_csv(skipped_path / f"{file_path.name}_analysis_results.csv")
                fixer = DocxGraphicsDataFixer()  # Reset for new extraction
                fixed_docx_path = Path(destination.as_posix().replace(".docx", "_FIXED.docx"))
                success = fixer.fix_docx_file(
                    input_path=destination.as_posix(),
                    output_path=fixed_docx_path,
                    method="extract",
                    extract_graphics_dir=skipped_path / "extracted_graphics"
                )
                if success:
                    print("✓ Fixed DOCX created with extracted graphics!")
                    summary = fixer.get_extraction_summary()
                    print(f"  Extracted {len(summary)} graphics files")
                    for path, size in summary:
                        print(f"    {path}: {size / 1024 / 1024:.1f} MB")
                    return True,fixed_docx_path

        except Exception as e:
            print(f"Error: {e}")

        return False, None

    def parse_file(self, file_path: Path) -> Tuple[bool, DoclingDocument]| tuple[bool, None]:
        parsing_success, doc = self._custom_parsing(file_path)
        if not parsing_success:
            result_processing, solved_docx_path = self.__process_corrupted_file(file_path)
            if result_processing:
                return self._custom_parsing(solved_docx_path)
        return parsing_success, doc

    @abstractmethod
    def _custom_parsing(self, file_path: Path) -> tuple[bool, DoclingDocument] | tuple[bool, None]:
        """
        Parse a single file and return its filename and markdown content.

        Args:
            filePath: Path to the file to parse

        Returns:
            str: markdown_content

        Raises:
            Exception: If parsing fails
        """
        pass

    def __export_to_json(self, doc:DoclingDocument) -> bool:
        json_output = doc.model_dump_json()
        #remove in doc.name substring "FIXED" if present
        name = doc.name.replace("_FIXED","")
        json_filename = self._parsed_docx_folder / f"{name}.json"
        try:
            with open(json_filename, "w", encoding='utf-8') as f:
                f.write(json_output)
        except Exception as e:
            print(f"Unable to write .json file: {json_filename}")
        return True

    def parse_all_files(self) -> List[str]:
        """
        Scan folder recursively and parse all files with the supported extension.

        Args:
            inputFolderPath: Path to the folder to scan
            outputFolderPath: Path to the output folder
        Returns:
            List[str]: files unable to be parsed
        """
        # Filter out temporary files (starting with ~$)
        print(f"Cuda is available: {torch.cuda.is_available()}")
        extract_stems = lambda paths: {Path(file_path).stem for file_path in paths}
        input_files = scan_folder(inputFolderPath=self.input_folder_path,extension=self.extension)
        already_parsed_files = extract_stems(scan_folder(inputFolderPath=self._parsed_docx_folder,extension=self.output_extension))
        new_files = {file for file in input_files if Path(file).stem not in already_parsed_files}
        print(f"Found new {len(new_files)} {self.extension} files to process")
        parsed_files = {}
        unparsed_files = []

        # Single tqdm progress bar
        for file_path in tqdm(new_files, desc="Parsing files", unit="file"):
            try:
                p = Path(file_path)
                # Update progress bar description dynamically
                tqdm.write(f"Parsing '{p.name}'")
                parsing_success, docx = self.parse_file(p)
                if parsing_success:
                    parsed_files[p.stem] = docx
                    output_file = self._parsed_docx_folder / f"{p.stem}.md"
                    #Save in markdown format for user
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(docx.export_to_markdown())
                    #Save in json format for Lossless serialization of Docling Document
                    self.__export_to_json(docx)
                    tqdm.write(f"Successfully parsed & saved: {p.name}")
                else:
                    raise Exception(f"Failed to parse '{p.name}'")

            except Exception as e:
                error_msg = str(e)[:50]
                tqdm.write(f"Error parsing {p.name}: {error_msg}")
                unparsed_files.append(p.as_posix())

        if unparsed_files:
            print(f"Failed to parse {len(unparsed_files)} files:")
            for file_path in unparsed_files:
                print(f"  - {file_path}")

        print(f"Successfully parsed {len(parsed_files)} out of {len(parsed_files)} files")
        return unparsed_files


class WordParser(BaseParser):
    """Parser for Microsoft Word documents (.docx files)."""

    def __init__(self, input_folder_path: str, output_folder_path: str, use_fallback: bool = False):
        """
        Initialize WordParser.

        Args:
            use_fallback: Whether to use mammoth as fallback if docling fails
        """
        super().__init__(input_folder_path, output_folder_path)
        self.use_fallback = use_fallback
        # Initialize converters
        try:
            self.converter = DocumentConverter()
            self.docling_available = True
        except ImportError:
            print("Warning: Docling not available.")
            self.docling_available = False
            self.converter = None

    @property
    def extension(self) -> str:
        return ".docx"

    def _custom_parsing(self, file_path: Path) -> tuple[bool, DoclingDocument] | tuple[bool, None]:
        """
        Parse a single .docx file and return its filename and markdown content.

        Args:
            filePath: Path to the .docx file to parse

        Returns:
            markdown_content

        Raises:
            Exception: If parsing fails with all available methods
        """
        filename = file_path.name

        # Try docling first if available
        if self.docling_available and self.converter:
            try:
                doc = self.converter.convert(file_path.as_posix()).document
                # Convert docling document to markdown
                return True,doc
            except Exception as e:
                docling_error = str(e)[:50]
                print(f"Failed to parse {filename}. Docling error for {filename}: {docling_error}")
        return False,None
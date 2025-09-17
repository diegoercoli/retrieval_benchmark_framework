import zipfile
import re
import base64
import os
from pathlib import Path
from typing import Union, List, Tuple, Optional
import logging


class DocxGraphicsDataFixer:
    """
    A class to fix DOCX files with extremely large o:gfxdata attributes that break XML parsers.

    The fixer can:
    1. Remove problematic graphics data entirely
    2. Extract graphics to separate files and replace with placeholders
    3. Truncate graphics data to manageable sizes
    """

    def __init__(self):
        self._setup_logging()
        self._extracted_graphics: List[Tuple[str, bytes]] = []

    def _setup_logging(self):
        """Set up logging for the fixer."""
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def fix_docx_file(self, input_path: Union[str, Path],
                      output_path: Union[str, Path],
                      method: str = "remove",
                      extract_graphics_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Fix a DOCX file by handling problematic graphics data.

        Args:
            input_path: Path to the problematic DOCX file
            output_path: Path where the fixed DOCX will be saved
            method: Fix method - "remove", "extract", or "truncate"
            extract_graphics_dir: Directory to save extracted graphics (for "extract" method)

        Returns:
            bool: True if successful, False otherwise
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            self._logger.error(f"Input file not found: {input_path}")
            return False

        self._logger.info(f"Fixing DOCX file: {input_path}")
        self._logger.info(f"Method: {method}")
        self._logger.info(f"Output: {output_path}")

        try:
            return self._process_docx_file(input_path, output_path, method, extract_graphics_dir)
        except Exception as e:
            self._logger.error(f"Error fixing DOCX file: {e}")
            return False

    def _process_docx_file(self, input_path: Path, output_path: Path,
                           method: str, extract_dir: Optional[Path]) -> bool:
        """Process the DOCX file with the specified method."""

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create extraction directory if needed
        if method == "extract" and extract_dir:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(input_path, 'r') as input_zip:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as output_zip:

                for file_info in input_zip.infolist():
                    file_content = input_zip.read(file_info.filename)

                    # Only process XML files that might contain graphics data
                    if file_info.filename.endswith('.xml') and self._might_contain_graphics(file_content):
                        self._logger.info(f"Processing: {file_info.filename}")

                        try:
                            # Decode and process XML content
                            xml_content = file_content.decode('utf-8', errors='replace')
                            fixed_content = self._fix_xml_content(
                                xml_content, file_info.filename, method, extract_dir
                            )
                            file_content = fixed_content.encode('utf-8')

                        except Exception as e:
                            self._logger.warning(f"Could not process {file_info.filename}: {e}")
                            # Keep original content if processing fails

                    # Write file to new archive
                    output_zip.writestr(file_info, file_content)

        self._logger.info(f"Fixed DOCX saved to: {output_path}")
        return True

    def _might_contain_graphics(self, content: bytes) -> bool:
        """Quick check if file might contain graphics data."""
        return b'o:gfxdata' in content or b'v:group' in content

    def _fix_xml_content(self, xml_content: str, filename: str, method: str,
                         extract_dir: Optional[Path]) -> str:
        """Fix XML content based on the selected method."""

        if method == "remove":
            return self._remove_graphics_data(xml_content, filename)
        elif method == "extract":
            return self._extract_graphics_data(xml_content, filename, extract_dir)
        elif method == "truncate":
            return self._truncate_graphics_data(xml_content, filename)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _remove_graphics_data(self, xml_content: str, filename: str) -> str:
        """Remove all o:gfxdata attributes entirely."""

        # Pattern to match o:gfxdata attributes with their values
        pattern = r'\s+o:gfxdata="[^"]*"'

        original_size = len(xml_content)
        fixed_content = re.sub(pattern, '', xml_content, flags=re.DOTALL)
        new_size = len(fixed_content)

        removed_mb = (original_size - new_size) / (1024 * 1024)
        self._logger.info(f"  Removed {removed_mb:.1f} MB of graphics data from {filename}")

        return fixed_content

    def _extract_graphics_data(self, xml_content: str, filename: str,
                               extract_dir: Path) -> str:
        """Extract graphics data to separate files and replace with placeholders."""

        if not extract_dir:
            self._logger.warning("No extract directory specified, falling back to remove method")
            return self._remove_graphics_data(xml_content, filename)

        def extract_and_replace(match):
            gfx_data = match.group(1)

            # Generate filename for extracted graphic
            graphic_id = len(self._extracted_graphics) + 1
            base_name = Path(filename).stem
            graphic_filename = f"{base_name}_graphic_{graphic_id}.b64"
            graphic_path = extract_dir / graphic_filename

            # Save the base64 data
            with open(graphic_path, 'w', encoding='utf-8') as f:
                f.write(gfx_data)

            self._extracted_graphics.append((str(graphic_path), gfx_data.encode()))

            # Try to decode and save as image if possible
            try:
                image_data = base64.b64decode(gfx_data)
                image_path = extract_dir / f"{base_name}_graphic_{graphic_id}.img"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                self._logger.info(f"  Extracted graphic to: {image_path}")
            except Exception:
                self._logger.info(f"  Extracted raw data to: {graphic_path}")

            # Return placeholder
            return f' o:gfxdata="EXTRACTED_TO_{graphic_filename}"'

        # Pattern to match o:gfxdata attributes
        pattern = r'\s+o:gfxdata="([^"]*)"'

        original_size = len(xml_content)
        fixed_content = re.sub(pattern, extract_and_replace, xml_content, flags=re.DOTALL)
        new_size = len(fixed_content)

        extracted_mb = (original_size - new_size) / (1024 * 1024)
        self._logger.info(f"  Extracted {extracted_mb:.1f} MB of graphics data from {filename}")

        return fixed_content

    def _truncate_graphics_data(self, xml_content: str, filename: str,
                                max_length: int = 1000) -> str:
        """Truncate graphics data to a manageable size."""

        def truncate_data(match):
            gfx_data = match.group(1)
            if len(gfx_data) > max_length:
                truncated = gfx_data[:max_length] + "...TRUNCATED"
                return f' o:gfxdata="{truncated}"'
            return match.group(0)

        # Pattern to match o:gfxdata attributes
        pattern = r'\s+o:gfxdata="([^"]*)"'

        original_size = len(xml_content)
        fixed_content = re.sub(pattern, truncate_data, xml_content, flags=re.DOTALL)
        new_size = len(fixed_content)

        truncated_mb = (original_size - new_size) / (1024 * 1024)
        self._logger.info(f"  Truncated {truncated_mb:.1f} MB of graphics data in {filename}")

        return fixed_content

    def get_extraction_summary(self) -> List[Tuple[str, int]]:
        """Get summary of extracted graphics."""
        return [(path, len(data)) for path, data in self._extracted_graphics]




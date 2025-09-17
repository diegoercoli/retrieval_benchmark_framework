import zipfile
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging


@dataclass
class AttributeProblem:
    """Data class to represent a problematic XML attribute."""
    line_number: int
    tag_name: str
    attribute_name: str
    attribute_length: int
    xml_file: str
    line_content_preview: str = ""


class DocxXMLAttributeAnalyzer:
    """
    A class for analyzing XML attributes in DOCX files to identify extremely long attribute values
    that cause parsing errors with lxml.

    This analyzer extracts XML files from DOCX archives and scans for attributes exceeding
    specified length thresholds, which commonly cause 'AttValue length too long' errors.
    """

    def __init__(self, max_length_threshold: int = 1000000):
        """
        Initialize the analyzer with a maximum attribute length threshold.

        Args:
            max_length_threshold (int): Attribute values longer than this will be flagged.
                                      Defaults to 1,000,000 characters (1MB).
        """
        self._max_length_threshold = max_length_threshold
        self._analysis_results: Dict[str, List[AttributeProblem]] = {}
        self._current_docx_path: Optional[Path] = None
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for the analyzer."""
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def analyze_docx_file(self, docx_path: Union[str, Path]) -> Dict[str, List[AttributeProblem]]:
        """
        Analyze all XML files within a DOCX file for problematic attributes.

        Args:
            docx_path (Union[str, Path]): Path to the DOCX file to analyze.

        Returns:
            Dict[str, List[AttributeProblem]]: Dictionary mapping XML file names to lists
                                             of problematic attributes found.

        Raises:
            FileNotFoundError: If the DOCX file does not exist.
            zipfile.BadZipFile: If the file is not a valid ZIP/DOCX archive.
        """
        self._current_docx_path = Path(docx_path)
        self._analysis_results.clear()

        if not self._current_docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        self._logger.info(f"Starting analysis of: {self._current_docx_path}")

        try:
            with zipfile.ZipFile(self._current_docx_path, "r") as zip_archive:
                xml_files = self._get_xml_files_from_archive(zip_archive)
                self._logger.info(f"Found {len(xml_files)} XML files to analyze")

                for xml_file in xml_files:
                    self._logger.debug(f"Analyzing: {xml_file}")
                    problems = self._analyze_xml_file(zip_archive, xml_file)
                    if problems:
                        self._analysis_results[xml_file] = problems
                        self._logger.info(f"Found {len(problems)} problems in {xml_file}")

        except zipfile.BadZipFile as e:
            raise zipfile.BadZipFile(f"Invalid DOCX/ZIP file: {docx_path}") from e

        total_problems = sum(len(problems) for problems in self._analysis_results.values())
        self._logger.info(
            f"Analysis complete. Found {total_problems} total problems in {len(self._analysis_results)} files")

        return self._analysis_results

    def get_analysis_summary(self) -> Dict[str, Union[int, float]]:
        """
        Get a summary of the analysis results.

        Returns:
            Dict[str, Union[int, float]]: Summary containing statistics about the analysis.
        """
        if not self._analysis_results:
            return {'files_with_problems': 0, 'total_problems': 0, 'average_problems_per_file': 0.0}

        total_problems = sum(len(problems) for problems in self._analysis_results.values())
        files_with_problems = len(self._analysis_results)

        # Find largest attribute
        max_attribute_size = 0
        for problems in self._analysis_results.values():
            for problem in problems:
                max_attribute_size = max(max_attribute_size, problem.attribute_length)

        return {
            'files_with_problems': files_with_problems,
            'total_problems': total_problems,
            'average_problems_per_file': total_problems / files_with_problems if files_with_problems > 0 else 0.0,
            'largest_attribute_size': max_attribute_size
        }

    def extract_context_around_problem(self, xml_file: str, line_number: int,
                                       context_lines: int = 5) -> str:
        """
        Extract XML context around a problematic attribute for inspection.

        Args:
            xml_file (str): Name of the XML file within the DOCX archive.
            line_number (int): Line number containing the problematic attribute.
            context_lines (int): Number of lines before and after to include.

        Returns:
            str: Formatted string containing the XML context with line numbers.

        Raises:
            ValueError: If no DOCX file has been analyzed yet.
        """
        if not self._current_docx_path:
            raise ValueError("No DOCX file has been analyzed yet")

        return self._extract_xml_context(xml_file, line_number, context_lines)

    def get_problematic_attributes_by_severity(self, min_length: int = 5000000) -> List[AttributeProblem]:
        """
        Get all problematic attributes above a certain severity threshold.

        Args:
            min_length (int): Minimum attribute length to consider severe.

        Returns:
            List[AttributeProblem]: List of severe problems sorted by attribute length (desc).
        """
        severe_problems = []

        for problems in self._analysis_results.values():
            for problem in problems:
                if problem.attribute_length >= min_length:
                    severe_problems.append(problem)

        return sorted(severe_problems, key=lambda p: p.attribute_length, reverse=True)

    def generate_report(self, include_context: bool = False) -> str:
        """
        Generate a comprehensive text report of the analysis results.

        Args:
            include_context (bool): Whether to include XML context for severe problems.

        Returns:
            str: Formatted report containing all analysis findings and recommendations.
        """
        if not self._analysis_results:
            return "No problematic attributes found or no analysis performed."

        report_lines = [
            "DOCX XML ATTRIBUTE ANALYSIS REPORT",
            "=" * 60,
            f"File analyzed: {self._current_docx_path}",
            f"Threshold: {self._max_length_threshold:,} characters",
            ""
        ]

        # Add summary first
        summary = self.get_analysis_summary()
        report_lines.extend([
            "SUMMARY:",
            f"- Files with problems: {summary['files_with_problems']}",
            f"- Total problematic attributes: {summary['total_problems']}",
            f"- Average problems per file: {summary['average_problems_per_file']:.1f}",
            f"- Largest attribute: {summary['largest_attribute_size']:,} characters",
            ""
        ])

        # Add detailed findings
        for xml_file, problems in sorted(self._analysis_results.items()):
            report_lines.extend([
                f"File: {xml_file}",
                "-" * 40
            ])

            # Sort problems by attribute length (largest first)
            sorted_problems = sorted(problems, key=lambda p: p.attribute_length, reverse=True)

            for problem in sorted_problems:
                size_mb = problem.attribute_length / (1024 * 1024)
                report_lines.append(
                    f"  Line {problem.line_number}: <{problem.tag_name}> "
                    f"attribute '{problem.attribute_name}' = {problem.attribute_length:,} chars ({size_mb:.1f} MB)"
                )

            report_lines.append("")

        # Show severe problems with context
        if include_context:
            severe_problems = self.get_problematic_attributes_by_severity(5000000)
            if severe_problems:
                report_lines.extend([
                    "MOST SEVERE PROBLEMS (with context):",
                    "-" * 40
                ])
                for i, problem in enumerate(severe_problems[:3], 1):  # Show top 3
                    size_mb = problem.attribute_length / (1024 * 1024)
                    report_lines.extend([
                        f"\n{i}. Problem: {problem.attribute_length:,} characters ({size_mb:.1f} MB) in {problem.xml_file}",
                        f"   Tag: <{problem.tag_name}>, Attribute: {problem.attribute_name}",
                    ])
                    try:
                        context = self.extract_context_around_problem(problem.xml_file, problem.line_number, 2)
                        report_lines.append("   Context:")
                        for line in context.split('\n'):
                            report_lines.append(f"   {line}")
                    except Exception as e:
                        report_lines.append(f"   Context extraction failed: {e}")
                    report_lines.append("")

        # Add recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "1. The file likely contains embedded objects, images, or corrupted data",
            "2. Try opening in Microsoft Word and saving as a new DOCX file",
            "3. Use mammoth library for text extraction: pip install mammoth",
            "4. Consider using python-docx or other specialized libraries",
            "5. For large embedded objects, extract them separately before parsing",
            "",
            "TECHNICAL NOTES:",
            "- lxml has a default limit for attribute values (~10MB)",
            "- Extremely long attributes often indicate base64-encoded content",
            "- This commonly occurs with embedded images or OLE objects"
        ])

        return '\n'.join(report_lines)

    def export_problems_to_csv(self, output_path: Union[str, Path]) -> None:
        """
        Export analysis results to a CSV file.

        Args:
            output_path (Union[str, Path]): Path for the output CSV file.
        """
        import csv

        output_path = Path(output_path)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['xml_file', 'line_number', 'tag_name', 'attribute_name', 'attribute_length', 'size_mb']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for xml_file, problems in self._analysis_results.items():
                for problem in problems:
                    writer.writerow({
                        'xml_file': problem.xml_file,
                        'line_number': problem.line_number,
                        'tag_name': problem.tag_name,
                        'attribute_name': problem.attribute_name,
                        'attribute_length': problem.attribute_length,
                        'size_mb': round(problem.attribute_length / (1024 * 1024), 2)
                    })

        self._logger.info(f"Results exported to: {output_path}")

    @property
    def max_length_threshold(self) -> int:
        """Get the current maximum length threshold."""
        return self._max_length_threshold

    @max_length_threshold.setter
    def max_length_threshold(self, value: int) -> None:
        """Set the maximum length threshold."""
        if value <= 0:
            raise ValueError("Threshold must be positive")
        self._max_length_threshold = value

    def _get_xml_files_from_archive(self, zip_archive: zipfile.ZipFile) -> List[str]:
        """
        Extract list of XML files from the ZIP archive.

        Args:
            zip_archive (zipfile.ZipFile): Open ZIP archive.

        Returns:
            List[str]: List of XML file names within the archive.
        """
        xml_files = [name for name in zip_archive.namelist() if name.endswith('.xml')]
        # Sort to ensure consistent processing order
        return sorted(xml_files)

    def _analyze_xml_file(self, zip_archive: zipfile.ZipFile, xml_file: str) -> List[AttributeProblem]:
        """
        Analyze a single XML file for problematic attributes.

        Args:
            zip_archive (zipfile.ZipFile): Open ZIP archive.
            xml_file (str): Name of XML file to analyze.

        Returns:
            List[AttributeProblem]: List of problems found in this XML file.
        """
        try:
            xml_content = zip_archive.read(xml_file).decode("utf-8", errors='replace')
            return self._scan_xml_content_for_long_attributes(xml_content, xml_file)
        except Exception as e:
            self._logger.warning(f"Could not analyze {xml_file}: {e}")
            return []

    def _scan_xml_content_for_long_attributes(self, xml_content: str, xml_file: str) -> List[AttributeProblem]:
        """
        Scan XML content for attributes exceeding the length threshold.

        Args:
            xml_content (str): Raw XML content as string.
            xml_file (str): Name of the XML file being analyzed.

        Returns:
            List[AttributeProblem]: List of problematic attributes found.
        """
        problems = []

        # Enhanced pattern to handle various attribute quote styles and namespaces
        # Matches both single and double quoted attributes
        patterns = [
            rf'(\w+(?::\w+)?)="([^"]{{{self._max_length_threshold},}})"',  # Double quotes
            rf"(\w+(?::\w+)?)='([^']{{{self._max_length_threshold},}})'",  # Single quotes
        ]

        for pattern in patterns:
            regex = re.compile(pattern, re.DOTALL | re.MULTILINE)

            for match in regex.finditer(xml_content):
                attr_name = match.group(1)
                attr_value = match.group(2)

                # Find the tag name by scanning backwards from attribute
                attr_start = match.start(1)
                tag_name = self._extract_tag_name_from_position(xml_content, attr_start)

                # Calculate line number
                line_number = xml_content[:match.start()].count('\n') + 1

                # Get line content preview (truncated)
                line_start = xml_content.rfind('\n', 0, match.start()) + 1
                line_end = xml_content.find('\n', match.end())
                if line_end == -1:
                    line_end = len(xml_content)
                line_content = xml_content[line_start:line_end]
                line_preview = line_content[:200] + "..." if len(line_content) > 200 else line_content

                problems.append(AttributeProblem(
                    line_number=line_number,
                    tag_name=tag_name,
                    attribute_name=attr_name,
                    attribute_length=len(attr_value),
                    xml_file=xml_file,
                    line_content_preview=line_preview
                ))

        return problems

    def _extract_tag_name_from_position(self, xml_content: str, attr_position: int) -> str:
        """
        Extract the tag name by looking backwards from an attribute position.

        Args:
            xml_content (str): The XML content.
            attr_position (int): Position of the attribute in the content.

        Returns:
            str: The tag name, or "UNKNOWN" if it cannot be determined.
        """
        # Look backwards for the opening < of the tag
        tag_start = xml_content.rfind('<', 0, attr_position)
        if tag_start == -1:
            return "UNKNOWN"

        # Find the end of the tag name (space, > or line ending)
        search_start = tag_start + 1
        tag_end = search_start

        # Skip any namespace prefix and find the actual tag name end
        while tag_end < len(xml_content) and xml_content[tag_end] not in ' \t\n\r>':
            tag_end += 1

        if tag_end > search_start:
            tag_name = xml_content[search_start:tag_end]
            # Handle self-closing tags and namespaces
            if tag_name.startswith('/'):
                tag_name = tag_name[1:]
            return tag_name

        return "UNKNOWN"

    def _extract_xml_context(self, xml_file: str, line_number: int, context_lines: int) -> str:
        """
        Extract XML context around a specific line number.

        Args:
            xml_file (str): XML file name within the DOCX.
            line_number (int): Target line number.
            context_lines (int): Number of context lines to include.

        Returns:
            str: Formatted context with line numbers.
        """
        try:
            with zipfile.ZipFile(self._current_docx_path, "r") as z:
                xml_content = z.read(xml_file).decode("utf-8", errors='replace')
                lines = xml_content.split('\n')

                start_line = max(0, line_number - context_lines - 1)
                end_line = min(len(lines), line_number + context_lines)

                context = []
                for i in range(start_line, end_line):
                    marker = " >>> " if i == line_number - 1 else "     "
                    # Truncate very long lines for readability, but show more for problematic lines
                    max_length = 500 if i == line_number - 1 else 200
                    line_content = lines[i][:max_length] + "..." if len(lines[i]) > max_length else lines[i]
                    context.append(f"{i + 1:5d}{marker}{line_content}")

                return '\n'.join(context)

        except Exception as e:
            return f"Error extracting context: {e}"

''' 
# Usage example
if __name__ == "__main__":
    # Example usage
    analyzer = DocxXMLAttributeAnalyzer(max_length_threshold=100000)  # 100KB threshold

    try:
        # Replace with your DOCX file path
        results = analyzer.analyze_docx_file("example.docx")

        # Generate and print report
        print(analyzer.generate_report(include_context=True))

        # Export to CSV if problems found
        if results:
            analyzer.export_problems_to_csv("docx_analysis_results.csv")

    except FileNotFoundError:
        print("Please provide a valid DOCX file path")
    except Exception as e:
        print(f"Analysis failed: {e}")
'''
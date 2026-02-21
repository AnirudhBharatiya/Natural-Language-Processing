import os
import unicodedata

class TextPreprocessor:
    """
    Handles NFKC normalization and punctuation separation.
    """
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def process_line(self, line):
        """Normalizes a single line of text."""
        text_data = unicodedata.normalize("NFKC", line)
        final_text = ""
        
        for i, char in enumerate(text_data):
            # Skip formatting characters
            if unicodedata.category(char) == "Cf":
                continue
                
            # Handle Punctuation
            if unicodedata.category(char).startswith("P"):
                if len(final_text) > 0 and final_text[-1] != ' ':
                    final_text += ' '
                
                final_text += char
                
                # Add space after punctuation if not already there
                if (i + 1 < len(text_data)) and (text_data[i + 1] != ' '):
                    final_text += ' '
            else:
                final_text += char
                
        return final_text.strip()

    def run(self):
        """Executes preprocessing on the input file."""
        dirname = os.path.dirname(self.output_file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        with open(self.input_file_path, "r", encoding="utf-8") as fin, \
             open(self.output_file_path, "w", encoding="utf-8") as fout:
            
            for line in fin:
                line = line.strip()
                if not line:
                    fout.write("\n")
                    continue
                    
                processed_line = self.process_line(line)
                fout.write(processed_line + "\n")
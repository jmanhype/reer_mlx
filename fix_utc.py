#!/usr/bin/env python3
import os
import re

for root, dirs, files in os.walk("."):
    if ".git" in root or "__pycache__" in root or ".mypy_cache" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath) as f:
                    content = f.read()
                if "from datetime import" in content and "timezone.utc" in content:
                    original = content
                    content = re.sub(
                        r"from datetime import (.*?)timezone(.*?)(?=\n)",
                        r"from datetime import \1timezone\2",
                        content,
                    )
                    content = content.replace("timezone,", "timezone,")
                    content = re.sub(r"\bUTC\b(?![\w])", "timezone.utc", content)
                    if content != original:
                        with open(filepath, "w") as f:
                            f.write(content)
                        print(f"Fixed: {filepath}")
            except:
                pass
print("Done!")

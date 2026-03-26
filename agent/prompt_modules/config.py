"""Auxiliary function prompt configs for Gemini agent."""

from typing import Tuple

CONFIG_AUXILIARY_FUNCTIONS = {
    "config0": """
AVAILABLE HIPPOCAMP AUXILIARY FUNCTIONS

INTERFACES:
NOTE: In Tool JSON, escape inner double quotes as \\\".

Function: return_metadata
Interface:
{
  "name": "return_metadata",
  "description": "Return file metadata for the given file path.",
  "applicable_when": "Need file attributes such as dates, type, modality, or location.",
  "pros": "Provides file attributes like creation/modification time and modality.",
  "cons": "Does not provide file content.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      }
    },
    "required": ["file_path"]
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"return_metadata \"PATH/TO/FILE.txt\""}}
Return Example:
{
  "success": true,
  "metadata": {
    "id": 113,
    "file_path": "PATH/TO/FILE.txt",
    "file_type": "txt",
    "file_modality": "text",
    "creation_date": "2025-01-01 10:00:00",
    "modification_date": "2025-11-09 21:35:00",
    "latitude": 1.3031,
    "longitude": 103.8317,
    "location": "The Orchard Residences, 238 Orchard Boulevard, Singapore 238858"
  },
  "error": null
}

Function: return_img
Interface:
{
  "name": "return_img",
  "description": "Return image conversion for the file at the given file path.",
  "applicable_when": "Need a rendered image or page view of a file.",
  "pros": "Useful for visual or scanned content and page-level inspection.",
  "cons": "Not ideal for purely text models and can be heavy to process.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      },
      "page": {
        "type": "int",
        "description": "Optional page number, passed as --page N."
      },
      "output_path": {
        "type": "str",
        "description": "Optional output path passed after file_path."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_img <file_path>`
- `return_img <file_path> <output_path>`
- `return_img <file_path> --page N`
- `return_img <file_path> <output_path> --page N`

Usage Example:
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\""}}
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.png\""}}
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\" --page 2"}}
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.png\" --page 2"}}
Return Example:
{
  "success": true,
  "image_path": "/hippocamp/output/FILE.png",
  "image_paths": [
    "/hippocamp/output/FILE.png"
  ],
  "image_b64": "iVBORw0KGgoAAAANSUhEUgAACWAAAAyACAIAAADHU68cAAAQAElEQVR4nOz9C1Rb553of29JlkBICAQy...",
  "image_b64_list": [
    "iVBORw0KGgoAAAANSUhEUgAACWAAAAyACAIAAADHU68cAAAQAElEQVR4nOz9C1Rb553of29JlkBICAQy..."
  ],
  "page_count": 1,
  "error": null
}

Function: return_ori
Interface:
{
  "name": "return_ori",
  "description": "Return the original file bytes for the given file path.",
  "applicable_when": "Need the original file bytes or a copy path.",
  "pros": "Provides the original file bytes for exact fidelity.",
  "cons": "Some models or tools cannot directly use raw bytes; confirm the file modality supports source-file transfer/parsing before using return_ori.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      },
      "output_path": {
        "type": "str",
        "description": "Optional output path passed after file_path."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_ori <file_path>`
- `return_ori <file_path> <output_path>`

Usage Example:
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\""}}
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.txt\""}}
Return Example:
{
  "success": true,
  "file_path": "/hippocamp/data/PATH/TO/FILE.txt",
  "file_b64": "CioqV2Vla2VuZCBUby1EbyBMaXN0KioKCipQcmlvcml0aWVzIGZvciBOb3YgOC05IFdlZWtlbmQ6Kgot...",
  "error": null
}

Function: return_txt
Interface:
{
  "name": "return_txt",
  "description": "Return the text version for the file at the given file path.",
  "applicable_when": "Need the text JSON representation of a file.",
  "pros": "Returns structured text version derived from the file.",
  "cons": "Output size depends on the file and may be large.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_txt <file_path>`

Usage Example:
{"name":"terminal","arguments":{"command":"return_txt \"PATH/TO/FILE.txt\""}}

Invalid Examples:
- `return_txt "x.pdf" --page 7` (invalid; `return_txt` does not support `--page`)
- `grep -i "..." "x.pdf"` (invalid; do not grep PDF directly, use `return_txt` first)
Return Example:
{
  "success": true,
  "data": {
    "file_info": {
      "id": 113,
      "user": "Victoria",
      "file_path": "PATH/TO/FILE.txt",
      "file_type": "txt",
      "file_name": "FILE.txt",
      "file_modality": "text",
      "creation_date": "2025-01-01 10:00:00",
      "modification_date": "2025-11-09 21:35:00",
      "latitude": "1.3031",
      "longitude": "103.8317",
      "location": "The Orchard Residences, 238 Orchard Boulevard, Singapore 238858",
      "QAID": "12 14 18",
      "QANum": "3.0"
    },
    "summary": "",
    "segments": [
      {
        "content": "\\n**Weekend To-Do List**\\n\\n*Priorities for Nov 8-9 Weekend:*\\n- [x] Guitar Research - check JustinGuitar welcome email, download first few videos. (Done Sat AM)\\n- [x] Redmart Delivery Sat 11am-1pm - Rece..."
      }
    ]
  },
  "error": null
}

Function: list_files
Interface:
{
  "name": "list_files",
  "description": "List files under /hippocamp/data.",
  "applicable_when": "Need a directory listing or file discovery.",
  "pros": "Fast overview of available files and folders.",
  "cons": "Does not show file contents or metadata.",
  "notes": "Only one pattern argument is allowed. If you need to parse the JSON output, pipe it (list_files | python/jq); do not use &&.",
  "arguments": {
    "type": "object",
    "properties": {
      "pattern": {
        "type": "str",
        "description": "Optional search pattern passed after the command."
      }
    },
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"list_files"}}
{"name":"terminal","arguments":{"command":"list_files \"*.pdf\""}}
Return Example:
{
  "success": true,
  "count": 11,
  "data": [
    "Documents/Notes/Note_Client_Dinner_Success_Sep2024.txt",
    "Documents/Notes/Note_Considering_Cat.txt",
    "Documents/Notes/Note_Interest_Watercolor.txt",
    "Documents/Notes/Note_Lily_Birthday_Plan_Nov2024.txt",
    "Documents/Notes/Note_MidTerm_Break_Final_Plan.txt",
    "Documents/Notes/Note_Next_Skill_Guitar.txt",
    "Documents/Notes/Note_Next_Skill_Tableau.txt",
    "Documents/Notes/Note_OctBreak_2023_Final_Plan.txt",
    "Documents/Notes/Note_Python_Homework_Reminder.txt",
    "Documents/Notes/Note_Watercolor_Goal_FramedPiece.txt",
    "PATH/TO/FILE.txt"
  ],
  "error": null
}

Function: hhelp
Interface:
{
  "name": "hhelp",
  "description": "Print the command help summary.",
  "applicable_when": "Need the command help summary.",
  "pros": "Summarizes available commands quickly.",
  "cons": "May be verbose and not task-specific.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"hhelp"}}
Return Example:
{
  "success": true,
  "data": "AVAILABLE COMMANDS:\n  return_txt <file_path>\n  return_img <file_path> [output_path] [--page N]\n  return_ori <file_path> [output_path]\n  return_metadata <file_path>\n  list_files [pattern]\n",
  "error": null
}

Function: webui
Interface:
{
  "name": "webui",
  "description": "Start the WebUI service.",
  "applicable_when": "Need to start the WebUI service.",
  "pros": "Provides a visual interface for browsing and monitoring.",
  "cons": "May require extra resources and is not needed for all tasks.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui"}}
Return Example:
{
  "success": true,
  "data": "WebUI started",
  "pid": "1858",
  "url": "http://localhost:8080",
  "log": "/hippocamp/output/.webui/webui.log",
  "error": null
}

Function: webui_stop
Interface:
{
  "name": "webui_stop",
  "description": "Stop the WebUI service.",
  "applicable_when": "Need to stop the WebUI service.",
  "pros": "Stops WebUI cleanly when it is not needed.",
  "cons": "Provides no additional task data.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_stop"}}
Return Example:
{
  "success": false,
  "data": "WebUI not running",
  "error": "WebUI not running"
}

Function: webui_status
Interface:
{
  "name": "webui_status",
  "description": "Check the WebUI service status.",
  "applicable_when": "Need to check whether WebUI is running.",
  "pros": "Quickly checks whether WebUI is running.",
  "cons": "Only reports status, not content.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_status"}}
Return Example:
{
  "success": false,
  "data": "WebUI is not running",
  "error": "not_running"
}
""".strip(),
    "config1": """
AVAILABLE HIPPOCAMP AUXILIARY FUNCTIONS

INTERFACES:

Function: return_metadata
Interface:
{
  "name": "return_metadata",
  "description": "Return file metadata for the given file path.",
  "applicable_when": "Need file attributes such as dates, type, modality, or location.",
  "pros": "Provides file attributes like creation/modification time and modality.",
  "cons": "Does not provide file content.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      }
    },
    "required": ["file_path"]
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"return_metadata \"PATH/TO/FILE.txt\""}}
Return Example:
{
  "success": true,
  "metadata": {
    "id": 113,
    "file_path": "PATH/TO/FILE.txt",
    "file_type": "txt",
    "file_modality": "text",
    "creation_date": "2025-01-01 10:00:00",
    "modification_date": "2025-11-09 21:35:00",
    "latitude": 1.3031,
    "longitude": 103.8317,
    "location": "The Orchard Residences, 238 Orchard Boulevard, Singapore 238858"
  },
  "error": null
}

Function: return_ori
Interface:
{
  "name": "return_ori",
  "description": "Return the original file bytes for the given file path.",
  "applicable_when": "Need the original file bytes or a copy path.",
  "pros": "Provides the original file bytes for exact fidelity.",
  "cons": "Some models or tools cannot directly use raw bytes; confirm the file modality supports source-file transfer/parsing before using return_ori.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      },
      "output_path": {
        "type": "str",
        "description": "Optional output path passed after file_path."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_ori <file_path>`
- `return_ori <file_path> <output_path>`

Usage Example:
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\""}}
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.txt\""}}
Return Example:
{
  "success": true,
  "file_path": "/hippocamp/data/PATH/TO/FILE.txt",
  "file_b64": "CioqV2Vla2VuZCBUby1EbyBMaXN0KioKCipQcmlvcml0aWVzIGZvciBOb3YgOC05IFdlZWtlbmQ6Kgot...",
  "error": null
}

Function: list_files
Interface:
{
  "name": "list_files",
  "description": "List files under /hippocamp/data.",
  "applicable_when": "Need a directory listing or file discovery.",
  "pros": "Fast overview of available files and folders.",
  "cons": "Does not show file contents or metadata.",
  "notes": "Only one pattern argument is allowed. If you need to parse the JSON output, pipe it (list_files | python/jq); do not use &&.",
  "arguments": {
    "type": "object",
    "properties": {
      "pattern": {
        "type": "str",
        "description": "Optional search pattern passed after the command."
      }
    },
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"list_files"}}
{"name":"terminal","arguments":{"command":"list_files \"*.pdf\""}}
Return Example:
{
  "success": true,
  "count": 11,
  "data": [
    "Documents/Notes/Note_Client_Dinner_Success_Sep2024.txt",
    "Documents/Notes/Note_Considering_Cat.txt",
    "Documents/Notes/Note_Interest_Watercolor.txt",
    "Documents/Notes/Note_Lily_Birthday_Plan_Nov2024.txt",
    "Documents/Notes/Note_MidTerm_Break_Final_Plan.txt",
    "Documents/Notes/Note_Next_Skill_Guitar.txt",
    "Documents/Notes/Note_Next_Skill_Tableau.txt",
    "Documents/Notes/Note_OctBreak_2023_Final_Plan.txt",
    "Documents/Notes/Note_Python_Homework_Reminder.txt",
    "Documents/Notes/Note_Watercolor_Goal_FramedPiece.txt",
    "PATH/TO/FILE.txt"
  ],
  "error": null
}

Function: hhelp
Interface:
{
  "name": "hhelp",
  "description": "Print the command help summary.",
  "applicable_when": "Need the command help summary.",
  "pros": "Summarizes available commands quickly.",
  "cons": "May be verbose and not task-specific.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"hhelp"}}
Return Example:
{
  "success": true,
  "data": "AVAILABLE COMMANDS:\n  return_txt <file_path>\n  return_img <file_path> [output_path] [--page N]\n  return_ori <file_path> [output_path]\n  return_metadata <file_path>\n  list_files [pattern]\n",
  "error": null
}

Function: webui
Interface:
{
  "name": "webui",
  "description": "Start the WebUI service.",
  "applicable_when": "Need to start the WebUI service.",
  "pros": "Provides a visual interface for browsing and monitoring.",
  "cons": "May require extra resources and is not needed for all tasks.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui"}}
Return Example:
{
  "success": true,
  "data": "WebUI started",
  "pid": "1858",
  "url": "http://localhost:8080",
  "log": "/hippocamp/output/.webui/webui.log",
  "error": null
}

Function: webui_stop
Interface:
{
  "name": "webui_stop",
  "description": "Stop the WebUI service.",
  "applicable_when": "Need to stop the WebUI service.",
  "pros": "Stops WebUI cleanly when it is not needed.",
  "cons": "Provides no additional task data.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_stop"}}
Return Example:
{
  "success": false,
  "data": "WebUI not running",
  "error": "WebUI not running"
}

Function: webui_status
Interface:
{
  "name": "webui_status",
  "description": "Check the WebUI service status.",
  "applicable_when": "Need to check whether WebUI is running.",
  "pros": "Quickly checks whether WebUI is running.",
  "cons": "Only reports status, not content.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_status"}}
Return Example:
{
  "success": false,
  "data": "WebUI is not running",
  "error": "not_running"
}
""".strip(),
    "config2": """
AVAILABLE HIPPOCAMP AUXILIARY FUNCTIONS

INTERFACES:

Function: return_metadata
Interface:
{
  "name": "return_metadata",
  "description": "Return file metadata for the given file path.",
  "applicable_when": "Need file attributes such as dates, type, modality, or location.",
  "pros": "Provides file attributes like creation/modification time and modality.",
  "cons": "Does not provide file content.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      }
    },
    "required": ["file_path"]
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"return_metadata \"PATH/TO/FILE.txt\""}}
Return Example:
{
  "success": true,
  "metadata": {
    "id": 113,
    "file_path": "PATH/TO/FILE.txt",
    "file_type": "txt",
    "file_modality": "text",
    "creation_date": "2025-01-01 10:00:00",
    "modification_date": "2025-11-09 21:35:00",
    "latitude": 1.3031,
    "longitude": 103.8317,
    "location": "The Orchard Residences, 238 Orchard Boulevard, Singapore 238858"
  },
  "error": null
}

Function: return_img
Interface:
{
  "name": "return_img",
  "description": "Return image conversion for the file at the given file path.",
  "applicable_when": "Need a rendered image or page view of a file.",
  "pros": "Useful for visual or scanned content and page-level inspection.",
  "cons": "Not ideal for purely text models and can be heavy to process.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      },
      "page": {
        "type": "int",
        "description": "Optional page number, passed as --page N."
      },
      "output_path": {
        "type": "str",
        "description": "Optional output path passed after file_path."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_img <file_path>`
- `return_img <file_path> <output_path>`
- `return_img <file_path> --page N`
- `return_img <file_path> <output_path> --page N`

Usage Example:
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\""}}
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.png\""}}
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\" --page 2"}}
{"name":"terminal","arguments":{"command":"return_img \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.png\" --page 2"}}
Return Example:
{
  "success": true,
  "image_path": "/hippocamp/output/FILE.png",
  "image_paths": [
    "/hippocamp/output/FILE.png"
  ],
  "image_b64": "iVBORw0KGgoAAAANSUhEUgAACWAAAAyACAIAAADHU68cAAAQAElEQVR4nOz9C1Rb553of29JlkBICAQy...",
  "image_b64_list": [
    "iVBORw0KGgoAAAANSUhEUgAACWAAAAyACAIAAADHU68cAAAQAElEQVR4nOz9C1Rb553of29JlkBICAQy..."
  ],
  "page_count": 1,
  "error": null
}

Function: return_ori
Interface:
{
  "name": "return_ori",
  "description": "Return the original file bytes for the given file path.",
  "applicable_when": "Need the original file bytes or a copy path.",
  "pros": "Provides the original file bytes for exact fidelity.",
  "cons": "Some models or tools cannot directly use raw bytes; confirm the file modality supports source-file transfer/parsing before using return_ori.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      },
      "output_path": {
        "type": "str",
        "description": "Optional output path passed after file_path."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_ori <file_path>`
- `return_ori <file_path> <output_path>`

Usage Example:
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\""}}
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.txt\""}}
Return Example:
{
  "success": true,
  "file_path": "/hippocamp/data/PATH/TO/FILE.txt",
  "file_b64": "CioqV2Vla2VuZCBUby1EbyBMaXN0KioKCipQcmlvcml0aWVzIGZvciBOb3YgOC05IFdlZWtlbmQ6Kgot...",
  "error": null
}

Function: list_files
Interface:
{
  "name": "list_files",
  "description": "List files under /hippocamp/data.",
  "applicable_when": "Need a directory listing or file discovery.",
  "pros": "Fast overview of available files and folders.",
  "cons": "Does not show file contents or metadata.",
  "notes": "Only one pattern argument is allowed. If you need to parse the JSON output, pipe it (list_files | python/jq); do not use &&.",
  "arguments": {
    "type": "object",
    "properties": {
      "pattern": {
        "type": "str",
        "description": "Optional search pattern passed after the command."
      }
    },
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"list_files"}}
{"name":"terminal","arguments":{"command":"list_files \"*.pdf\""}}
Return Example:
{
  "success": true,
  "count": 11,
  "data": [
    "Documents/Notes/Note_Client_Dinner_Success_Sep2024.txt",
    "Documents/Notes/Note_Considering_Cat.txt",
    "Documents/Notes/Note_Interest_Watercolor.txt",
    "Documents/Notes/Note_Lily_Birthday_Plan_Nov2024.txt",
    "Documents/Notes/Note_MidTerm_Break_Final_Plan.txt",
    "Documents/Notes/Note_Next_Skill_Guitar.txt",
    "Documents/Notes/Note_Next_Skill_Tableau.txt",
    "Documents/Notes/Note_OctBreak_2023_Final_Plan.txt",
    "Documents/Notes/Note_Python_Homework_Reminder.txt",
    "Documents/Notes/Note_Watercolor_Goal_FramedPiece.txt",
    "PATH/TO/FILE.txt"
  ],
  "error": null
}

Function: hhelp
Interface:
{
  "name": "hhelp",
  "description": "Print the command help summary.",
  "applicable_when": "Need the command help summary.",
  "pros": "Summarizes available commands quickly.",
  "cons": "May be verbose and not task-specific.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"hhelp"}}
Return Example:
{
  "success": true,
  "data": "AVAILABLE COMMANDS:\n  return_txt <file_path>\n  return_img <file_path> [output_path] [--page N]\n  return_ori <file_path> [output_path]\n  return_metadata <file_path>\n  list_files [pattern]\n",
  "error": null
}

Function: webui
Interface:
{
  "name": "webui",
  "description": "Start the WebUI service.",
  "applicable_when": "Need to start the WebUI service.",
  "pros": "Provides a visual interface for browsing and monitoring.",
  "cons": "May require extra resources and is not needed for all tasks.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui"}}
Return Example:
{
  "success": true,
  "data": "WebUI started",
  "pid": "1858",
  "url": "http://localhost:8080",
  "log": "/hippocamp/output/.webui/webui.log",
  "error": null
}

Function: webui_stop
Interface:
{
  "name": "webui_stop",
  "description": "Stop the WebUI service.",
  "applicable_when": "Need to stop the WebUI service.",
  "pros": "Stops WebUI cleanly when it is not needed.",
  "cons": "Provides no additional task data.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_stop"}}
Return Example:
{
  "success": false,
  "data": "WebUI not running",
  "error": "WebUI not running"
}

Function: webui_status
Interface:
{
  "name": "webui_status",
  "description": "Check the WebUI service status.",
  "applicable_when": "Need to check whether WebUI is running.",
  "pros": "Quickly checks whether WebUI is running.",
  "cons": "Only reports status, not content.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_status"}}
Return Example:
{
  "success": false,
  "data": "WebUI is not running",
  "error": "not_running"
}
""".strip(),
    "config3": """
AVAILABLE HIPPOCAMP AUXILIARY FUNCTIONS

INTERFACES:

Function: return_metadata
Interface:
{
  "name": "return_metadata",
  "description": "Return file metadata for the given file path.",
  "applicable_when": "Need file attributes such as dates, type, modality, or location.",
  "pros": "Provides file attributes like creation/modification time and modality.",
  "cons": "Does not provide file content.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      }
    },
    "required": ["file_path"]
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"return_metadata \"PATH/TO/FILE.txt\""}}
Return Example:
{
  "success": true,
  "metadata": {
    "id": 113,
    "file_path": "PATH/TO/FILE.txt",
    "file_type": "txt",
    "file_modality": "text",
    "creation_date": "2025-01-01 10:00:00",
    "modification_date": "2025-11-09 21:35:00",
    "latitude": 1.3031,
    "longitude": 103.8317,
    "location": "The Orchard Residences, 238 Orchard Boulevard, Singapore 238858"
  },
  "error": null
}

Function: return_ori
Interface:
{
  "name": "return_ori",
  "description": "Return the original file bytes for the given file path.",
  "applicable_when": "Need the original file bytes or a copy path.",
  "pros": "Provides the original file bytes for exact fidelity.",
  "cons": "Some models or tools cannot directly use raw bytes; confirm the file modality supports source-file transfer/parsing before using return_ori.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      },
      "output_path": {
        "type": "str",
        "description": "Optional output path passed after file_path."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_ori <file_path>`
- `return_ori <file_path> <output_path>`

Usage Example:
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\""}}
{"name":"terminal","arguments":{"command":"return_ori \"PATH/TO/FILE.txt\" \"/hippocamp/output/FILE.txt\""}}
Return Example:
{
  "success": true,
  "file_path": "/hippocamp/data/PATH/TO/FILE.txt",
  "file_b64": "CioqV2Vla2VuZCBUby1EbyBMaXN0KioKCipQcmlvcml0aWVzIGZvciBOb3YgOC05IFdlZWtlbmQ6Kgot...",
  "error": null
}

Function: return_txt
Interface:
{
  "name": "return_txt",
  "description": "Return the text JSON for the file at the given file path.",
  "applicable_when": "Need the text JSON representation of a file.",
  "pros": "Returns structured text JSON derived from the file.",
  "cons": "Output size depends on the file and may be large.",
  "arguments": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "str",
        "description": "The file path under /hippocamp/data, passed after the command."
      }
    },
    "required": ["file_path"]
  }
}
Recommended Command Forms:
- `return_txt <file_path>`

Usage Example:
{"name":"terminal","arguments":{"command":"return_txt \"PATH/TO/FILE.txt\""}}

Invalid Examples:
- `return_txt "x.pdf" --page 7` (invalid; `return_txt` does not support `--page`)

Return Example:
{
  "success": true,
  "data": {
    "file_info": {
      "id": 113,
      "user": "Victoria",
      "file_path": "PATH/TO/FILE.txt",
      "file_type": "txt",
      "file_name": "FILE.txt",
      "file_modality": "text",
      "creation_date": "2025-01-01 10:00:00",
      "modification_date": "2025-11-09 21:35:00",
      "latitude": "1.3031",
      "longitude": "103.8317",
      "location": "The Orchard Residences, 238 Orchard Boulevard, Singapore 238858",
      "QAID": "12 14 18",
      "QANum": "3.0"
    },
    "summary": "",
    "segments": [
      {
        "content": "\\n**Weekend To-Do List**\\n\\n*Priorities for Nov 8-9 Weekend:*\\n- [x] Guitar Research - check JustinGuitar welcome email, download first few videos. (Done Sat AM)\\n- [x] Redmart Delivery Sat 11am-1pm - Rece..."
      }
    ]
  },
  "error": null
}

Function: list_files
Interface:
{
  "name": "list_files",
  "description": "List files under /hippocamp/data.",
  "applicable_when": "Need a directory listing or file discovery.",
  "pros": "Fast overview of available files and folders.",
  "cons": "Does not show file contents or metadata.",
  "notes": "Only one pattern argument is allowed. If you need to parse the JSON output, pipe it (list_files | python/jq); do not use &&.",
  "arguments": {
    "type": "object",
    "properties": {
      "pattern": {
        "type": "str",
        "description": "Optional search pattern passed after the command."
      }
    },
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"list_files"}}
{"name":"terminal","arguments":{"command":"list_files \"*.pdf\""}}
Return Example:
{
  "success": true,
  "count": 11,
  "data": [
    "Documents/Notes/Note_Client_Dinner_Success_Sep2024.txt",
    "Documents/Notes/Note_Considering_Cat.txt",
    "Documents/Notes/Note_Interest_Watercolor.txt",
    "Documents/Notes/Note_Lily_Birthday_Plan_Nov2024.txt",
    "Documents/Notes/Note_MidTerm_Break_Final_Plan.txt",
    "Documents/Notes/Note_Next_Skill_Guitar.txt",
    "Documents/Notes/Note_Next_Skill_Tableau.txt",
    "Documents/Notes/Note_OctBreak_2023_Final_Plan.txt",
    "Documents/Notes/Note_Python_Homework_Reminder.txt",
    "Documents/Notes/Note_Watercolor_Goal_FramedPiece.txt",
    "PATH/TO/FILE.txt"
  ],
  "error": null
}

Function: hhelp
Interface:
{
  "name": "hhelp",
  "description": "Print the command help summary.",
  "applicable_when": "Need the command help summary.",
  "pros": "Summarizes available commands quickly.",
  "cons": "May be verbose and not task-specific.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"hhelp"}}
Return Example:
{
  "success": true,
  "data": "AVAILABLE COMMANDS:\n  return_txt <file_path>\n  return_img <file_path> [output_path] [--page N]\n  return_ori <file_path> [output_path]\n  return_metadata <file_path>\n  list_files [pattern]\n",
  "error": null
}

Function: webui
Interface:
{
  "name": "webui",
  "description": "Start the WebUI service.",
  "applicable_when": "Need to start the WebUI service.",
  "pros": "Provides a visual interface for browsing and monitoring.",
  "cons": "May require extra resources and is not needed for all tasks.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui"}}
Return Example:
{
  "success": true,
  "data": "WebUI started",
  "pid": "1858",
  "url": "http://localhost:8080",
  "log": "/hippocamp/output/.webui/webui.log",
  "error": null
}

Function: webui_stop
Interface:
{
  "name": "webui_stop",
  "description": "Stop the WebUI service.",
  "applicable_when": "Need to stop the WebUI service.",
  "pros": "Stops WebUI cleanly when it is not needed.",
  "cons": "Provides no additional task data.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_stop"}}
Return Example:
{
  "success": false,
  "data": "WebUI not running",
  "error": "WebUI not running"
}

Function: webui_status
Interface:
{
  "name": "webui_status",
  "description": "Check the WebUI service status.",
  "applicable_when": "Need to check whether WebUI is running.",
  "pros": "Quickly checks whether WebUI is running.",
  "cons": "Only reports status, not content.",
  "arguments": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
Usage Example:
{"name":"terminal","arguments":{"command":"webui_status"}}
Return Example:
{
  "success": false,
  "data": "WebUI is not running",
  "error": "not_running"
}
""".strip(),
}


AVAILABLE_CONFIGS: Tuple[str, ...] = tuple(CONFIG_AUXILIARY_FUNCTIONS.keys())


def get_auxiliary_functions_block(config_name: str) -> str:
    key = (config_name or "").strip().lower()
    if key not in CONFIG_AUXILIARY_FUNCTIONS:
        valid = ", ".join(AVAILABLE_CONFIGS)
        raise ValueError(f"Unknown prompt config: {config_name!r}. Available: {valid}")
    return CONFIG_AUXILIARY_FUNCTIONS[key]

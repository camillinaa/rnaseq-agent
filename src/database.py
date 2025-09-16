import logging
import sqlite3
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RNAseqDatabase:
    """Handle SQLite database operations for RNAseq data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.connect()

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info("Database connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        if not self.connection:
            if not self.connect():
                return {"error": "Database connection failed"}

        try:
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
            if any(keyword in query.upper() for keyword in dangerous_keywords):
                return {"error": "Only SELECT queries are allowed"}

            self.connection.row_factory = sqlite3.Row
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]

            return {
                "success": True,
                "data": results,
                "columns": columns,
                "row_count": len(results)
            }
        except Exception as e:
            return {"error": f"Query execution failed: {str(e)}"}

    def get_table_names(self) -> List[str]:
        """Return a list of all table names in the connected SQLite database."""
        if not self.connection:
            if not self.connect():
                return []
        try:
            self.connection.row_factory = None
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            table_names = [row[0] for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(table_names)} table names from SQLite")
            return table_names
        except Exception as e:
            logger.error(f"Error fetching table names: {e}")
            return []
            
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about available tables and their schemas"""
        if not self.connection:
            if not self.connect():
                return {"error": "Database connection failed"}
        try:
            self.connection.row_factory = None
            tables = self.get_table_names()
            table_info = {}
            for table in tables:
                pragma_query = f"PRAGMA table_info({table});"
                cursor = self.connection.cursor()
                cursor.execute(pragma_query)
                columns = cursor.fetchall()

                table_info[table] = {
                    "columns": [{"name": col[1], "type": col[2]} for col in columns],
                    "sample_query": f"SELECT * FROM {table} LIMIT 5;"
                }
            return {"success": True, "tables": table_info}
        except Exception as e:
            return {"error": f"Failed to get table info: {str(e)}"}

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
#!/usr/bin/env python3
"""
twscrape Account Setup Guide and Helper Script
Sets up X/Twitter authentication for data collection
"""

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()


async def setup_twscrape():
    """Interactive setup for twscrape accounts"""

    console.print("\n[bold cyan]🔐 twscrape X/Twitter Account Setup[/bold cyan]\n")

    console.print(
        Panel(
            """
[yellow]⚠️  IMPORTANT DISCLAIMER:[/yellow]
    
1. This tool requires X/Twitter account credentials
2. Use dedicated accounts, NOT your personal account
3. X may suspend accounts used for scraping
4. Consider X's Terms of Service for your use case
5. For production, use official X API instead

[red]By proceeding, you acknowledge these risks.[/red]
    """,
            title="Legal Notice",
            border_style="red",
        )
    )

    if not Confirm.ask("\n[yellow]Do you want to proceed with setup?[/yellow]"):
        console.print("[red]Setup cancelled.[/red]")
        return

    console.print("\n[cyan]Choose setup method:[/cyan]")
    console.print("1. Username/Password authentication")
    console.print("2. Cookie-based authentication (more reliable)")
    console.print("3. Import existing session")

    choice = Prompt.ask("Select option", choices=["1", "2", "3"])

    if choice == "1":
        await setup_password_auth()
    elif choice == "2":
        await setup_cookie_auth()
    else:
        await import_session()


async def setup_password_auth():
    """Setup using username/password"""

    console.print("\n[yellow]📝 Username/Password Setup[/yellow]\n")

    # Create accounts file
    accounts_file = Path("accounts.txt")

    console.print("Enter account details (one per line):")
    console.print("Format: username:password:email:email_password")
    console.print("Example: mybot:pass123:bot@email.com:emailpass123")
    console.print("\n[dim]Press Ctrl+D when done[/dim]\n")

    accounts = []
    try:
        while True:
            line = input()
            if line.strip():
                accounts.append(line.strip())
    except EOFError:
        pass

    if accounts:
        with open(accounts_file, "w") as f:
            f.write("\n".join(accounts))

        console.print(
            f"\n[green]✅ Saved {len(accounts)} account(s) to {accounts_file}[/green]"
        )

        # Now add accounts to twscrape
        console.print("\n[yellow]Adding accounts to twscrape...[/yellow]")

        try:
            from twscrape import API

            api = API()

            # Add accounts
            await api.pool.add_account(
                username=accounts[0].split(":")[0],
                password=accounts[0].split(":")[1],
                email=(
                    accounts[0].split(":")[2]
                    if len(accounts[0].split(":")) > 2
                    else None
                ),
                email_password=(
                    accounts[0].split(":")[3]
                    if len(accounts[0].split(":")) > 3
                    else None
                ),
            )

            # Login
            console.print("[yellow]Attempting login...[/yellow]")
            await api.pool.login_all()

            console.print("[green]✅ Account added and logged in![/green]")

        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            console.print("\n[yellow]Manual setup required:[/yellow]")
            console.print("Run these commands:")
            console.print("[dim]twscrape add_accounts accounts.txt")
            console.print("twscrape login_accounts[/dim]")


async def setup_cookie_auth():
    """Setup using browser cookies (more reliable)"""

    console.print("\n[yellow]🍪 Cookie-Based Setup (Recommended)[/yellow]\n")

    instructions = """
[cyan]Step 1: Get your X/Twitter cookies[/cyan]
1. Open X.com in your browser
2. Log in to the account you want to use
3. Open Developer Tools (F12)
4. Go to Application/Storage → Cookies
5. Find and copy these cookie values:
   - auth_token
   - ct0 (CSRF token)

[cyan]Step 2: Create cookies file[/cyan]
Create a file 'cookies.json' with this format:
"""

    console.print(Panel(instructions, border_style="cyan"))

    cookie_template = """{
  "accounts": [
    {
      "username": "your_username",
      "auth_token": "your_auth_token_here",
      "ct0": "your_ct0_token_here"
    }
  ]
}"""

    console.print("[yellow]Cookie template:[/yellow]")
    console.print(Panel(cookie_template, border_style="dim"))

    # Create template file
    if Confirm.ask("\n[yellow]Create template cookies.json file?[/yellow]"):
        with open("cookies.json", "w") as f:
            f.write(cookie_template)
        console.print("[green]✅ Created cookies.json template[/green]")
        console.print("[yellow]Edit this file with your actual cookie values[/yellow]")

    console.print("\n[cyan]Step 3: Import cookies to twscrape[/cyan]")
    console.print("After editing cookies.json, run:")
    console.print("[dim]python setup_twscrape.py --import-cookies[/dim]")


async def import_session():
    """Import existing session database"""

    console.print("\n[yellow]📦 Import Existing Session[/yellow]\n")

    session_path = Prompt.ask("Enter path to existing session database")

    if Path(session_path).exists():
        console.print(f"[green]✅ Found session database at {session_path}[/green]")
        console.print("\nUse this in your collection commands:")
        console.print(f"[dim]--session-db {session_path}[/dim]")
    else:
        console.print(f"[red]❌ Session database not found at {session_path}[/red]")


async def test_connection():
    """Test if accounts are working"""

    console.print("\n[yellow]🧪 Testing twscrape connection...[/yellow]\n")

    try:
        from twscrape import API

        api = API()

        # Check for accounts
        accounts = await api.pool.get_all()

        if not accounts:
            console.print("[red]❌ No accounts found in database[/red]")
            console.print("Run setup first")
            return

        console.print(f"[green]✅ Found {len(accounts)} account(s)[/green]")

        # Try a simple search
        console.print("\n[yellow]Testing search functionality...[/yellow]")

        test_query = "python"
        results = []
        async for tweet in api.search(test_query, limit=1):
            results.append(tweet)

        if results:
            console.print(
                f"[green]✅ Successfully retrieved {len(results)} tweet(s)![/green]"
            )
            console.print(
                "\n[bold green]Setup is working! You can now collect data.[/bold green]"
            )
        else:
            console.print(
                "[yellow]⚠️  No results returned (account may need verification)[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("1. Check account credentials")
        console.print("2. Account may be suspended/locked")
        console.print("3. Try using cookie authentication instead")


async def import_cookies_from_file():
    """Import cookies from JSON file"""

    import json

    console.print("\n[yellow]📥 Importing cookies from file...[/yellow]\n")

    try:
        with open("cookies.json") as f:
            data = json.load(f)

        from twscrape import API

        api = API()

        for account in data.get("accounts", []):
            console.print(f"Adding account: {account['username']}")

            # Add account with cookies directly
            cookies_dict = {"auth_token": account["auth_token"], "ct0": account["ct0"]}

            # Direct SQL insertion for cookie-based account
            import sqlite3

            conn = sqlite3.connect("accounts.db")
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    email TEXT,
                    email_password TEXT,
                    user_agent TEXT,
                    cookies TEXT,
                    active INTEGER DEFAULT 1,
                    locks TEXT DEFAULT '[]',
                    stats TEXT DEFAULT '{}'
                )
            """
            )

            import json

            user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            cursor.execute(
                """
                INSERT OR REPLACE INTO accounts (username, password, email, email_password, user_agent, cookies, active)
                VALUES (?, '', '', '', ?, ?, 1)
            """,
                (account["username"], user_agent, json.dumps(cookies_dict)),
            )

            conn.commit()
            conn.close()

        console.print("[green]✅ Cookies imported successfully![/green]")

    except FileNotFoundError:
        console.print("[red]❌ cookies.json not found[/red]")
    except Exception as e:
        console.print(f"[red]❌ Import failed: {e}[/red]")


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(description="twscrape setup helper")
    parser.add_argument("--test", action="store_true", help="Test existing setup")
    parser.add_argument(
        "--import-cookies", action="store_true", help="Import cookies from cookies.json"
    )

    args = parser.parse_args()

    if args.test:
        asyncio.run(test_connection())
    elif args.import_cookies:
        asyncio.run(import_cookies_from_file())
    else:
        asyncio.run(setup_twscrape())


if __name__ == "__main__":
    main()

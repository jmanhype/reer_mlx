# twscrape Authentication Setup Guide

## ⚠️ Important Notice
- **Use dedicated bot accounts, NOT personal accounts**
- **X/Twitter may suspend accounts used for scraping**
- **For production use, consider official X API**
- **Review X's Terms of Service for compliance**

## Quick Setup Options

### Option 1: Cookie Authentication (Most Reliable) 🍪

This method uses browser cookies from a logged-in session:

1. **Get Cookies from Browser:**
   ```
   1. Log into X.com in your browser
   2. Open Developer Tools (F12)
   3. Go to Application → Cookies → https://x.com
   4. Find these values:
      - auth_token (long string)
      - ct0 (CSRF token)
   ```

2. **Create cookies.json:**
   ```json
   {
     "accounts": [
       {
         "username": "your_username",
         "auth_token": "paste_auth_token_here",
         "ct0": "paste_ct0_token_here"
       }
     ]
   }
   ```

3. **Import to twscrape:**
   ```bash
   python setup_twscrape.py --import-cookies
   ```

### Option 2: Username/Password Authentication

1. **Create accounts.txt:**
   ```
   username1:password1:email@example.com:emailpassword
   username2:password2:email2@example.com:emailpass2
   ```

2. **Add accounts:**
   ```bash
   twscrape add_accounts accounts.txt
   twscrape login_accounts
   ```

### Option 3: Interactive Setup

Run the helper script for guided setup:
```bash
python setup_twscrape.py
```

## Testing Your Setup

```bash
# Test if authentication is working
python setup_twscrape.py --test

# Or try a manual collection
python scripts/social_collect.py collect x \
  -k "test" \
  --limit 5
```

## Troubleshooting

### Common Issues:

1. **"No active accounts"**
   - Accounts not logged in properly
   - Try cookie authentication instead

2. **Rate Limiting**
   - Use multiple accounts
   - Add delays between requests
   - Reduce collection limits

3. **Account Suspended**
   - X detected automation
   - Use different account
   - Consider official API

### Session Database Location

twscrape stores sessions in SQLite database:
- Default: `accounts.db` in current directory
- Custom: Use `--session-db path/to/db.sqlite`

## Using with Collection Script

Once authenticated:

```bash
# Search tweets
python scripts/social_collect.py collect x \
  -k "machine learning" \
  --min-likes 100 \
  --days-back 7 \
  --limit 200

# User timeline
python scripts/social_collect.py collect x \
  --user elonmusk \
  --include-replies \
  --limit 50

# With custom session DB
python scripts/social_collect.py collect x \
  -k "AI" \
  --session-db /path/to/sessions.sqlite
```

## Security Best Practices

1. **Never commit credentials:**
   ```bash
   # Add to .gitignore
   echo "accounts.txt" >> .gitignore
   echo "cookies.json" >> .gitignore
   echo "*.sqlite" >> .gitignore
   echo "accounts.db" >> .gitignore
   ```

2. **Use environment variables:**
   ```bash
   export TWSCRAPE_DB=/secure/path/sessions.sqlite
   ```

3. **Rotate accounts regularly**

4. **Monitor account health**

## Alternative: Official X API

For production/compliance:
1. Apply for X API access at developer.twitter.com
2. Use Tweepy or official SDK
3. Costs money but fully compliant
4. Higher rate limits

## Resources

- [twscrape Documentation](https://github.com/vladkens/twscrape)
- [X Developer Portal](https://developer.twitter.com)
- [X API v2 Documentation](https://developer.twitter.com/en/docs/twitter-api)
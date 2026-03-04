# Dokku Deployment with Persistent Storage

## Problem
SQLite database is stored in `/bot/data/bot.db` inside the container, but without persistent storage it gets deleted on every deployment/restart.

## Solution
Create a persistent storage mount in Dokku to preserve the database between deployments.

## Setup Commands

```bash
# 1. Create persistent storage directory on host
dokku storage:ensure-directory insta-bot

# 2. Mount the storage to container's /bot/data directory
dokku storage:mount insta-bot /var/lib/dokku/data/storage/insta-bot:/bot/data

# 3. Verify the mount
dokku storage:report insta-bot

# 4. Rebuild and restart the app
dokku ps:rebuild insta-bot
```

## Verify It Works

After deployment, check the logs:
```bash
dokku logs insta-bot -t
```

You should see:
```
Database initialized at data/bot.db
```

Then test by:
1. Send: `ботяра, привіт`
2. Send: `ботяра, який мій попередній запит?`
3. Bot should remember the conversation

After restart:
```bash
dokku ps:restart insta-bot
```

The conversation context should persist.

## Check Database File

```bash
# SSH into the container
dokku enter insta-bot web

# Check if database exists
ls -lh /bot/data/
cat /bot/data/bot.db  # Should show binary data

# Exit container
exit
```

## Troubleshooting

### Database not persisting
```bash
# Check if mount exists
dokku storage:report insta-bot

# Should show:
# Storage mount:  /var/lib/dokku/data/storage/insta-bot:/bot/data
```

### Permission issues
```bash
# Fix permissions on host
sudo chown -R dokku:dokku /var/lib/dokku/data/storage/insta-bot
sudo chmod -R 755 /var/lib/dokku/data/storage/insta-bot
```

### Check logs for database operations
```bash
# Enable DEBUG logging
dokku config:set insta-bot LOG_LEVEL=DEBUG

# Watch logs
dokku logs insta-bot -t
```

Look for:
- `Loading user data from database for user_id: XXX`
- `Found user data in database: context=X messages`
- `Saving user data to database: user_id=XXX`

## Backup Database

```bash
# Backup
sudo cp /var/lib/dokku/data/storage/insta-bot/bot.db /var/lib/dokku/data/storage/insta-bot/bot.db.backup

# Restore
sudo cp /var/lib/dokku/data/storage/insta-bot/bot.db.backup /var/lib/dokku/data/storage/insta-bot/bot.db
dokku ps:restart insta-bot
```

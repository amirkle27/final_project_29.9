"""Streamlit admin dashboard for users, tokens, and usage logs."""

import streamlit as st
import pandas as pd
import sqlite3
import dal

DB_PATH = dal.DB_NAME


def _conn():
    """Open a SQLite connection to the project DB."""
    return sqlite3.connect(DB_PATH)


@st.cache_data(ttl=10)
def load_users(search: str = "") -> pd.DataFrame:
    """Return users table (optionally filtered by username substring)."""
    base = "SELECT username, tokens, joined_at, usage_count FROM users"
    params = []
    if search:
        base += " WHERE username LIKE ?"
        params.append(f"%{search}%")
    with _conn() as conn:
        return pd.read_sql_query(base, conn, params=params)


@st.cache_data(ttl=10)
def load_usage(username: str | None = None, limit: int = 100) -> pd.DataFrame:
    """Return recent usage logs; filter by username and limit rows."""
    if username:
        q = """
            SELECT created_at, username, action, model_name, file_name, tokens_after_usage
            FROM usage_logs
            WHERE username=?
            ORDER BY datetime(created_at) DESC
            LIMIT ?
        """
        params = [username, limit]
    else:
        q = """
            SELECT created_at, username, action, model_name, file_name, tokens_after_usage
            FROM usage_logs
            ORDER BY datetime(created_at) DESC
            LIMIT ?
        """
        params = [limit]
    with _conn() as conn:
        return pd.read_sql_query(q, conn, params=params)


def add_tokens(username: str, amount: int) -> tuple[bool, str | int]:
    """Add tokens to a user via DAL; returns (ok, new_total or error message)."""
    u = dal.get_user(username)
    if not u:
        return False, "User not found"
    new_total = int(u["tokens"]) + int(amount)
    dal.update_tokens(username, new_total)
    # Record in usage logs
    dal.log_usage(username, "admin:add_tokens", None, None, tokens_after_usage=new_total)
    return True, new_total


def main():
    """Render the Streamlit admin UI."""
    st.set_page_config(page_title="ML Server Admin", layout="wide")
    st.title("📊 User Tokens Dashboard (Streamlit)")
    st.caption("Direct reads from SQLite per project requirements; can be extended later to API + JWT.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("👥 Users")
        search = st.text_input("Search by username", "")
        users_df = load_users(search)
        st.dataframe(users_df, use_container_width=True)

        if st.button("🔄 Refresh table"):
            st.cache_data.clear()

    with right:
        st.subheader("➕ Add tokens")
        all_users = load_users()["username"].tolist()
        if all_users:
            sel_user = st.selectbox("Select user", all_users, index=0)
            amt = st.number_input("How many tokens to add?", min_value=1, max_value=1000, value=10, step=1)
            if st.button("✅ Add tokens"):
                ok, res = add_tokens(sel_user, int(amt))
                if ok:
                    st.success(f"Added {amt} tokens to {sel_user}. New total: {res}")
                    st.cache_data.clear()
                else:
                    st.error(str(res))
        else:
            st.info("No users to display.")

    with st.expander("🪵 Usage Logs (recent)"):
        f_user = st.text_input("Filter by username (optional)", "")
        limit = st.slider("How many rows to show?", min_value=10, max_value=500, value=100, step=10)
        logs_df = load_usage(username=f_user or None, limit=limit)
        st.dataframe(logs_df, use_container_width=True)


if __name__ == "__main__":
    dal.init_db()
    main()

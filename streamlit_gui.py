import streamlit as st
import pandas as pd
import sqlite3
import dal

DB_PATH = dal.DB_NAME

def _conn():
    return sqlite3.connect(DB_PATH)

@st.cache_data(ttl=10)
def load_users(search: str = "") -> pd.DataFrame:
    base = "SELECT username, tokens, joined_at, usage_count FROM users"
    params = []
    if search:
        base += " WHERE username LIKE ?"
        params.append(f"%{search}%")
    with _conn() as conn:
        return pd.read_sql_query(base, conn, params=params)

@st.cache_data(ttl=10)
def load_usage(username: str | None = None, limit: int = 100) -> pd.DataFrame:
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
    """מוסיף טוקנים למשתמש דרך פונקציות ה-DAL כדי לשמור עקביות ולוגים."""
    u = dal.get_user(username)
    if not u:
        return False, "User not found"
    new_total = int(u["tokens"]) + int(amount)
    dal.update_tokens(username, new_total)  # עדכון tokens בטבלת users:contentReference[oaicite:7]{index=7}
    dal.log_usage(username, "admin:add_tokens", None, None, tokens_after_usage=new_total)  # רישום ללוגים:contentReference[oaicite:8]{index=8}
    return True, new_total

def main():
    st.set_page_config(page_title="ML Server Admin", layout="wide")
    st.title("📊 User Tokens Dashboard (Streamlit)")
    st.caption("קריאה ישירה מ-SQLite לפי דרישות הפרויקט; אפשר להרחיב בעתיד ל-API + JWT.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("👥 משתמשים")
        search = st.text_input("חיפוש לפי שם משתמש", "")
        users_df = load_users(search)
        st.dataframe(users_df, use_container_width=True)

        if st.button("🔄 רענן טבלה"):
            st.cache_data.clear()

    with right:
        st.subheader("➕ הוספת טוקנים")
        all_users = load_users()["username"].tolist()
        if all_users:
            sel_user = st.selectbox("בחר משתמש", all_users, index=0)
            amt = st.number_input("כמה טוקנים להוסיף?", min_value=1, max_value=1000, value=10, step=1)
            if st.button("✅ הוסף טוקנים"):
                ok, res = add_tokens(sel_user, int(amt))
                if ok:
                    st.success(f"התווספו {amt} טוקנים ל-{sel_user}. כעת יש לו: {res}")
                    st.cache_data.clear()
                else:
                    st.error(str(res))
        else:
            st.info("אין משתמשים להצגה.")

    with st.expander("🪵 Usage Logs (אחרונים)"):
        f_user = st.text_input("סינון לפי משתמש (לא חובה)", "")
        limit = st.slider("כמה שורות להציג?", min_value=10, max_value=500, value=100, step=10)
        logs_df = load_usage(username=f_user or None, limit=limit)
        st.dataframe(logs_df, use_container_width=True)

if __name__ == "__main__":
    # לוודא שהטבלאות קיימות, בדיוק כמו בשרת:contentReference[oaicite:9]{index=9}
    dal.init_db()
    main()
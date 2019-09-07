"""
a caribou migration

name: create_samples_table 
version: 20190907154856
"""

def upgrade(connection):
    sql = """
    CREATE TABLE samples (
        id     INTEGER PRIMARY KEY ASC,
        source TEXT
    )
    """
    connection.execute(sql)
    pass

def downgrade(connection):
    connection.execute('DROP TABLE samples')
    pass

"""
a caribou migration

name: add_path_to_samples 
version: 20190907160059
"""

def upgrade(connection):
    connection.execute("""
      ALTER TABLE samples ADD COLUMN path TEXT
    """)
    pass

def downgrade(connection):
    # TODO: sqlite has no simple column deletion support
    # looks like I have to create a new table without the given column
    # and copy the data?! christ
    pass

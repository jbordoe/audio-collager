"""
a caribou migration

name: add_features_to_samples 
version: 20190907164418
"""

def upgrade(connection):
    connection.execute("""
      ALTER TABLE samples ADD COLUMN features TEXT
    """)
    pass

def downgrade(connection):
    # TODO: sqlite has no simple column deletion support
    # looks like I have to create a new table without the given column
    # and copy the data?! christ
    pass

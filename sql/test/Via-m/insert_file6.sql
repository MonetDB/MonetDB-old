select add_file((select media_id
                 from   media
                 where  identifier = 'identifier1'),
                        'fabchannel2007',
                        'filename 3',
                        'codec 1',
                        2000,
                        1,
                        'url 1',
                        'empty',
                        250,
                        250);

select add_file((select media_id
                 from   media
                 where  identifier = 'identifier2'),
                        'fabchannel2007',
                        'filename 2',
                        'codec 1',
                        2000,
                        2,
                        'url 1',
                        'empty',
                        250,
                        250);

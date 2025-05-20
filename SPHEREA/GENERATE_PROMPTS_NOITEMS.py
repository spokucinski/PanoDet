def generate_prompts(room_types, styles, day_times):
    """
    room_types - a list of strings with desired room types, e.g. living_room (no spaces)
    styles - a dictionary with lists as values (key is a room name compatible with names from room_types), 
                e.g. styles = {'living_room': ['contemporary','mid-century modern']}
    day_times - a list of day times, e.g. 'day', 'night'
    """
    for idx, room_type in enumerate(room_types):
        for style_name in styles[room_types[idx]]:
            for day_time in day_times:
                print(f"I want you to generate a flat, rectangular equirectangular panorama image with a 2:1 aspect ratio. Try to place the camera in the middle of the room to capture the surroundings in full. Do not place a camera in one corner - camera's position in the middle of the room is typical. The image should have a 2:1 aspect ratio, provide a full 360-degree horizontal and 180-degree vertical field of view, suitable for a virtual tour viewer. After transforming the panorama to a 360-degree view, it should be continuous. It also should include a typical equirectangular distortion and some camera noise, but if you cannot fully include this technical attribute required for seamless integration into virtual tour software, do what you can to make it as similar to reality as possible. The equirectangular panorama should cover the whole image space (it is a must, so do not include any frames or padding). The equirectangular panorama depicts a {room_type.replace('_', ' ')} at {day_time} decorated in a {style_name} style with the typical items of furniture. Remember that a typical room has some kind of a door. Please try to create a typical room, so focus on typicality of different furniture items in a given room type and also on the typicality of each item's context. Do not put too many objects of the same type in the scene.")
                print()
                
if __name__ == '__main__':   
    styles = {
    'living_room': [
        'contemporary',
        'mid-century modern',
        'bohemian',
        'industrial',
        'scandinavian',
        'farmhouse',
        'traditional',
        'eclectic',
        'coastal',
        'minimalist',
        'typical middle-class family living in Europe',
        'typical working-class family living in Europe',
        'vintage industrial',
        'urban loft',
        'art nouveau'
    ],
    'bedroom': [
        'romantic',
        'modern glam',
        'rustic',
        'japanese zen',
        'vintage',
        'shabby chic',
        'art deco',
        'mediterranean',
        'transitional',
        'boho chic',
        'typical middle-class family living in Europe',
        'typical working-class family living in Europe',
        'scandinavian minimalist',
        'cozy cottage',
        'urban contemporary'
    ],
    'kitchen': [
        'modern farmhouse',
        'industrial loft',
        'french country',
        'mediterranean',
        'scandinavian',
        'minimalist',
        'coastal',
        'retro',
        'traditional',
        'eclectic',
        'typical middle-class family living in Europe',
        'typical working-class family living in Europe',
        'vintage industrial',
        'bohemian chic',
        'contemporary sleek'
    ],
    'home_office': [
        'modern minimalist',
        'scandinavian',
        'industrial chic',
        'vintage retro',
        'mid-century modern',
        'traditional executive',
        'bohemian',
        'eclectic',
        'zen-inspired',
        'contemporary',
        'typical middle-class family living in Europe',
        'typical working-class family living in Europe',
        'minimalist urban',
        'vintage eclectic',
        'cozy rustic'
    ],
    'bathroom': [
        'spa-inspired',
        'coastal retreat',
        'industrial chic',
        'scandinavian',
        'classic white',
        'vintage glam',
        'nature-inspired',
        'modern minimalist',
        'mediterranean',
        'tropical paradise',
        'typical middle-class family living in Europe',
        'typical working-class family living in Europe',
        'elegant contemporary',
        'vintage chic',
        'modern urban'
        
        ]
    }

    room_types = ['living_room', 'bedroom', 'kitchen', 'home_office', 'bathroom']

    day_times = ['day', 'night']
    
    generate_prompts(room_types, styles, day_times)